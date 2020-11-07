import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torchtext
from transformers import *

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
from bert import BERTClassifier
from lstm import ValDataset

from tqdm import tqdm

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_input_array(sentences, tokenizer, max_seq_len=None):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for sent in tqdm(sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True)  # Construct attn. masks.
        # return_tensors='tf',  # Return tf tensors.
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    token_type_ids = np.array(token_type_ids)
    return [input_ids, attention_masks, token_type_ids]

def encode_sentence(text, vocab2index, N=250):
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    if length > 0:
        return encoded, length


def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.shape
    return nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix)), embedding_dim


def make_weight_matrix(target_vocab):
    glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus
                                  dim=100)  # embedding size = 100
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 100))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(100,))
    print(type(weights_matrix))
    return weights_matrix


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
    convert_to_np = lambda x: x.cpu().detach().numpy()  # convert to numpy on cpu
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


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # set seed and any other hyper-parameters
    random_seed = 42
    train_test_split_ratio = 0.2
    train_val_split_ratio = .1
    loss = 'FL'  # 'CE', 'FL', 'WCE'
    epochs = 50
    batch_size = 32
    model_type = 'BERT'  # 'BERT'

    # read data
    FOLDER_PATH = "so_dataset"
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned_rm_stopw.csv'))
    q_bodies = df['body'].apply(lambda x: x.split('|'))
    q_titles = df['title'].apply(lambda x: x.split('|'))
    q_tags = df['tag_list'].apply(lambda x: x.split('|'))
    labels = df['label']

    if model_type != 'BERT':
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

        assert len(words) == len(vocab2index)
        print(f"Vocab size: {len(vocab2index)}")

        # TODO: assign features
        # q_bodies.append(q_titles)
        # q_bodies.append(q_tags)

        X = q_titles.apply(lambda x: np.array(encode_sentence(x, vocab2index)))
    else:
        tokenizer = AutoTokenizer.from_pretrained("lanwuwei/BERTOverflow_stackoverflow_github")
        X = create_input_array(q_titles, tokenizer, max_seq_len=250)

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
        label_weights = torch.Tensor(class_weights).to(device)  # make torch tensor
        criterion = nn.CrossEntropyLoss(weight=label_weights)
    elif loss == 'FL':
        criterion = FocalLoss(alpha=0.6, gamma=2, smooth=1e-5)
    else:
        criterion = nn.CrossEntropyLoss()

    embedding, embedding_dim = create_emb_layer(make_weight_matrix(words))

    if model_type == 'LSTM':
        model = LSTM(embedding=embedding, emb_dim=embedding_dim, dimension=256, num_layers=2)
    elif model_type == 'BERT':
        model = AutoModelForTokenClassification.from_pretrained("lanwuwei/BERTOverflow_stackoverflow_github")
        # model = BERTClassifier()
    return
    train_model(model, train_dl, val_dl, test_dl, epochs=epochs, lr=0.01, criterion=criterion)


if __name__ == '__main__':
    run()