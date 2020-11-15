import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torchtext

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
from lstm import LSTM
from lstm import ValDataset

from tqdm import tqdm

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_sentence(text, vocab2index, N=64):
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    if length > 0:
        return encoded, length    
    
def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.shape
    return nn.Embedding.from_pretrained(weights_matrix), embedding_dim

def make_weight_matrix(target_vocab):
    glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=100)   # embedding size = 100
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 100))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
    return weights_matrix

def train_model(model, train_dl, valid_dl, test_dl, epochs=30, lr=0.001, criterion=None):
    print('here')
    print(USE_GPU)
    print('here1')
    model.to(USE_GPU)
    print('here2')
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
        if(val_acc >.35):
            break
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
    convert_to_np = lambda x: x.cpu().detach().numpy() # convert to numpy on cpu
    for x, y, l in valid_dl:
        # if test_data:
        #     print(torch.max(x), torch.min(x))
        x, y = x.long().to(USE_GPU), y.long().to(USE_GPU)
        y_hat = model(x, l)
        loss = criterion(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        y_pred.extend(convert_to_np(pred))
        y_true.extend(convert_to_np(y))
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.unsqueeze(-1).cpu())) * y.cpu().shape[0]
    if test_data:
        get_metrics(y_pred=y_pred, y_true=y_true, save_dir="./", model_name='lstm')
    else:
        return sum_loss / total, correct / total, sum_rmse / total


def calculate_class_weights(labels, version='sklearn'):
    if version == 'sklearn':
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    elif version == 'probs':
        class_count = np.unique(labels, return_counts=True)[1]
        class_weights = 1. / class_count
    else:
        # https://forums.fast.ai/t/correcting-class-imbalance-for-nlp/22152/6
        counts = Counter(labels)
        trn_weights = [count / len(labels) for idx, count in counts.items()]
        class_weights = np.array([max(trn_weights) / value for value in trn_weights])
    return class_weights  # make weights out of inverse counts

def run(max_length = 64, dim = 256, layer_num = 2, alpha = .01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # set seed and any other hyper-parameters
    random_seed = 42
    train_test_split_ratio = 0.2
    train_val_split_ratio = .1
    loss = 'WCE'  # 'CE', 'FL', 'WCE'
    epochs = 30
    batch_size = 32

    # read data
    FOLDER_PATH = "so_dataset"
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned_rm_stopw.csv'))
    q_bodies = df['body'].apply(lambda x: x.split('|'))
    q_titles = df['title'].apply(lambda x: x.split('|'))
    q_tags = df['tag_list'].apply(lambda x: x.split('|'))
    labels = df['label']		

    counts = Counter()
    for rows in q_titles:
        counts.update([r.strip().lower() for r in rows if r.strip() != ""])

    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    i = 2
    for word in counts:
        vocab2index[word] = i
        words.append(word)
        i += 1

    # TODO: assign features
    q_bodies.append(q_titles)
    q_bodies.append(q_tags)

    X = q_bodies.apply(lambda x: np.array(encode_sentence(x, vocab2index, N = max_length)))
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
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if loss == 'WCE':
        class_weights = calculate_class_weights(labels, version='sklearn')  # make class-weight
        label_weights = class_weights  # make torch tensor
        criterion = nn.CrossEntropyLoss(weight=label_weights)
    elif loss == 'FL':
        criterion = FocalLoss(alpha=0.6, gamma=2, smooth=1e-5)
    else:
        criterion = nn.CrossEntropyLoss()

    embedding, embedding_dim = create_emb_layer(make_weight_matrix(words))

    model = LSTM(embedding = embedding, emb_dim=embedding_dim, dimension=dim, num_layers=layer_num)
    assert len(words) == len(vocab2index)
    print(f"Vocab size: {len(vocab2index)}")
    train_model(model, train_dl, val_dl, test_dl, epochs=epochs, lr=alpha, criterion=criterion)

if __name__ == '__main__':
    lr = [.0001, .001, .01]
    dim = [128, 256, 512]
    num_layers = [2, 3, 4]
    max_length = [32, 64, 128]
    for alpha in lr:
        for d in dim:
            for layer in num_layers:
                for length in max_length:
                    print('test')
                    print("learning rate " + str(alpha) + " dimensions " + str(d) + " layers " + str(layer) + " max length " + str(length))
                    run(max_length = length, dim = d, layer_num = layer, alpha = alpha)
