import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

from model_metrics import get_metrics
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, Dataset
# import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from lossess.focal_loss import FocalLoss
from sklearn.utils import class_weight

'''
from tqdm.notebook import tqdm
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
'''

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):

    def __init__(self, leng, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(leng, 300, padding_idx=0)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2 * dimension, 5)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)  # .cuda()  # throw in raw text

        # TODO: debug
        # import code
        # code.interact(local={**locals(), **globals()})

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # better to pack the text

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        # text_fea = torch.squeeze(text_fea, 1)
        # text_out = torch.sigmoid(text_fea)

        return text_fea


class ValDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X.iloc[idx][0].astype(np.int32)), self.y.iloc[idx], self.X.iloc[idx][1]


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False).cuda()
        out_pack, (ht, ct) = self.lstm(x_pack).cuda()
        out = self.linear(ht[-1]).cuda()
        return out


def encode_sentence(text, vocab2index, N=250):
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    if length > 0:
        return encoded, length


def train_model(model, train_dl, valid_dl, test_dl, epochs=10, lr=0.001, criterion=None):
    print(USE_GPU)
    model.to(USE_GPU)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if criterion is None:
        print("Cannot Train Model if Loss is None")
        return
    optimizer = torch.optim.Adam(parameters, lr=lr)
    print("Training model...")
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x, y = x.long().to(USE_GPU), y.long().to(USE_GPU)
            optimizer.zero_grad()
            y_pred = model(x, l)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, valid_dl, criterion=criterion)
        # if i % 5 == 1:
        print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
            sum_loss / total, val_loss, val_acc, val_rmse))
    validation_metrics(model, test_dl, test_data=True, criterion=criterion)


def validation_metrics(model, valid_dl, test_data=False, criterion=None):
    if criterion is None:
        print("Cannot Eval Model if Loss is None")
        return
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    y_pred = []
    y_true = []
    for x, y, l in valid_dl:
        if test_data:
            print(torch.max(x), torch.min(x))
            # print(x.shape, y.shape, l)
        x, y = x.long().to(USE_GPU), y.long().to(USE_GPU)
        y_hat = model(x, l)
        loss = criterion(y_hat, y)
        pred = torch.max(y_hat, 1)[1]  # .cuda()
        y_pred.append(pred)
        y_true.append(y)
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.unsqueeze(-1).cpu())) * y.cpu().shape[0]
    if test_data:
        get_metrics(y_pred=y_pred, y_true=y_true, save_dir="./", model_name='lstm')
    else:
        return sum_loss / total, correct / total, sum_rmse / total


# def pred(model, test_dl):
#     model.cuda()
#     model.eval()
#     y_pred = []
#     y_true = []
#     for x, y, l in test_dl:
#         x = x.long() # .cuda()
#         l = l.long() # .cuda()
#         y = y.long() # .cuda()
#         y_hat = model(x, l)
#         pred = torch.max(y_hat, 1)[1].cuda()
#         y_pred.append(pred)
#         y_true.append(y)
#     print(y_pred)
#     print(y_true)
#     get_metrics(y_pred=y_pred, y_true=y_true, save_dir="./", model_name='lstm')


def lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # set seed and any other hyper-parameters
    random_seed = 42
    train_test_split_ratio = 0.2
    train_val_split_ratio = .1
    loss = 'CE'  # 'FL', 'WCE'

    # read data
    FOLDER_PATH = "so_dataset"
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    q_bodies = df['body'].apply(lambda x: x.split('|'))
    q_titles = df['title'].apply(lambda x: x.split('|'))
    q_tags = df['tag_list'].apply(lambda x: x.split('|'))
    labels = df['label']

    counts = Counter()
    for rows in q_bodies:
        counts.update([r.strip().lower() for r in rows if r != ""])

    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    i = 2
    for word in counts:
        vocab2index[word] = i
        words.append(word)
        i += 1

    # TODO: assign features
    # q_bodies.append(q_titles)
    # q_bodies.append(q_tags)

    X = q_bodies.apply(lambda x: np.array(encode_sentence(x, vocab2index)))
    y = labels

    # train-val-test split
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=train_test_split_ratio,
                                                              random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=train_val_split_ratio,
                                                      random_state=random_seed)

    X_train.reset_index(drop=True)
    X_val.reset_index(drop=True)
    X_test.reset_index(drop=True)

    y_train.reset_index(drop=True)
    y_val.reset_index(drop=True)
    y_test.reset_index(drop=True)

    train_ds = ValDataset(X_train, y_train)
    valid_ds = ValDataset(X_val, y_val)
    test_ds = ValDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

    if loss == 'WCE':
        # TODO: Add other weighting formulas
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)  # make class-weight
        label_weights = torch.Tensor(class_weights).to(device)  # make torch tensor
        criterion = nn.CrossEntropyLoss(weight=label_weights)
    elif loss == 'FL':
        # TODO: add value for alpha and other
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model = LSTM(len(vocab2index))
    print(f"Vocab size {len(vocab2index), len(words)}")
    import code
    code.interact(local={**locals(), **globals()})
    # TODO: set epoch > 1
    train_model(model, train_dl, val_dl, test_dl, epochs=0, lr=0.01, criterion=criterion)


if __name__ == '__main__':
    lstm()
