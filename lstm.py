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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
'''
from tqdm.notebook import tqdm
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
'''
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

class LSTM_variable_input(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        
    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

def encode_sentence(text, vocab2index):
    encoded = np.zeros(len(vocab2index), dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    encoded = enc1
    return encoded, len(encoded)

def train_model(model, train_dl, valid_dl, epochs=10, lr=0.001,):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def pred (model, valid_dl):
    model.eval()
    y_pred = []
    y_true = []
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        y_pred.append(y_hat)
        y_true.append(y)
    get_metrics(y_pred=y_pred, y_true=y_true, save_dir="./", model_name='lstm')

def lstm():
    # set seed and any other hyper-parameters
    random_seed = 42
    train_test_split_ratio = 0.4
    train_val_split_ratio = .5

    # read data
    FOLDER_PATH = "so_dataset"
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    q_bodies = df['body'].apply(lambda x: x.split('|'))
    q_titles = df['title'].apply(lambda x: x.split('|'))
    q_tags = df['tag_list'].apply(lambda x: x.split('|'))
    labels = df['label']

    counts = Counter()
    for rows in q_bodies:
        counts.update(rows)
    print("num_words before:",len(counts.keys()))


    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    # assign features

    X = q_bodies.apply(lambda x: np.array(encode_sentence(x,vocab2index )))
    print(X)
    y = df['label']

    # train-val-test split
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=train_test_split_ratio,
                                                        random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=train_val_split_ratio,
                                                        random_state=random_seed)

    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, shuffle=True)
    val_dl = DataLoader(valid_ds)

    model = LSTM_variable_input(len(vocab2index), 50, 50)
    train_model(model, train_dl, val_dl, epochs=30, lr=0.1)

if __name__ == '__main__':
    lstm()