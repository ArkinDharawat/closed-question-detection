import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torchtext
from transformers import *
import argparse

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
from lstm import ValDataset, BERTDataset

from tqdm import tqdm

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_array(x):
    return np.transpose(np.stack(x, axis=0), (1, 0, 2))


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
    path_to_params = "./bert_model_params"
    model.to(USE_GPU)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if criterion is None:
        print("Cannot Train Model if Loss is None")
        return
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    print("Training model...")
    max_acc = 0.0
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0
        for x, y, l in train_dl:
            x, y = x.long().to(USE_GPU), y.long().to(USE_GPU)
            optimizer.zero_grad()  # zero params
            y_pred = model(x, l)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            # print(f"loss so far: {sum_loss / total}")
        # import code
        # code.interact(local={**locals(), **globals()})

        val_loss, val_acc = validation_metrics(model, valid_dl, criterion=criterion)
        # if max_acc < val_acc:
        #     max_acc = val_acc
        #
        # if i % 5 == 1:
        print("train loss %.3f, val loss %.3f, val accuracy %.3f, and train accuracy %.3f" % (
            sum_loss / total, val_loss, val_acc, correct / total))
    print("Testing model...")

    validation_metrics(model, test_dl, test_data=True, criterion=criterion)


def validation_metrics(model, dl_iter, test_data=False, criterion=None):
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

    def convert_to_np(arr):
        return arr.cpu().detach().numpy()  # convert to numpy on cpu

    with torch.no_grad():
        for x, y, l in dl_iter:
            # if test_data:
            #     print(torch.max(x), torch.min(x))
            x, y = x.long().to(USE_GPU), y.long().to(USE_GPU)
            y_hat = model(x, l)
            # check the output values for each batch
            # import code
            # code.interact(local={**locals(), **globals()})
            loss = criterion(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            y_pred.extend(convert_to_np(pred))
            y_true.extend(convert_to_np(y))
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item() * y.shape[0]
    if test_data:
        import code
        code.interact(local={**locals(), **globals()})
        get_metrics(y_pred=y_pred, y_true=y_true, save_dir="./", model_name='bert')
    else:
        return sum_loss / total, correct / total,


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
    parser = argparse.ArgumentParser(description='train model')
    # set seed and any other hyper-parameters
    train_test_split_ratio = 0.2
    train_val_split_ratio = .1
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--loss', type=str, default="CE", help='Should be WCE, CE or FL')
    parser.add_argument('--epochs', type=int, default=25, help='epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batches to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of model')
    parser.add_argument('--model', type=str, default="LSTM", help='Should be BERT or LSTM')
    args = parser.parse_args()

    random_seed = args.seed
    loss = args.loss
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_type = args.model

    # set seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # read data
    FOLDER_PATH = "so_dataset"
    # so_questions_cleaned_rm_stopw.csv
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
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
    elif model_type == "BERT":
        X = q_titles + q_bodies + q_titles

    y = labels

    # train-val-test split
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=train_test_split_ratio,
                                                              random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=train_val_split_ratio,
                                                      random_state=random_seed)

    if model_type == "BERT":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # select train mini-batch, loss decreases
        max_lenght = 64 # shorter length
        X_train = create_input_array(X_train, tokenizer, max_seq_len=max_lenght)
        X_test = create_input_array(X_test, tokenizer, max_seq_len=max_lenght)
        X_val = create_input_array(X_val, tokenizer, max_seq_len=max_lenght)

        train_ds = BERTDataset(transform_array(X_train), y_train)
        valid_ds = BERTDataset(transform_array(X_val), y_val)
        test_ds = BERTDataset(transform_array(X_test), y_test)
        # batch_size = min(batch_size, 32)  # smaller size for BERT
        # import code
        # code.interact(local={**locals(), **globals()})

    elif model_type == "LSTM":
        X_train.reset_index(drop=True)
        X_val.reset_index(drop=True)
        X_test.reset_index(drop=True)

        y_train.reset_index(drop=True)
        y_val.reset_index(drop=True)
        y_test.reset_index(drop=True)

        train_ds = ValDataset(X_train, y_train)
        valid_ds = ValDataset(X_val, y_val)
        test_ds = ValDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if loss == 'WCE':
        # sklearn
        class_weights = calculate_class_weights(labels, version='sklearn')  # make class-weight
        label_weights = torch.Tensor(class_weights).to(device)  # make torch tensor
        print(f"Weights are {label_weights}")
        criterion = nn.CrossEntropyLoss(weight=label_weights)
    elif loss == 'FL':
        # alpha = 0.6 doesn't seem to work
        # alpha as class weights
        # gamma = 0.5 -> FL = .34, 10 epochs
        # gamma = 1 -> FL = .33, 15 epochs
        # gamma = 2 -> F1 = .29, 10 epochs
        # gamma = 5 -> F1 = .31, 10 epochs
        class_weights = calculate_class_weights(labels, version='sklearn')  # make class-weight
        criterion = FocalLoss(alpha=class_weights, gamma=5, smooth=1e-5)
    else:
        criterion = nn.CrossEntropyLoss()

    if model_type == 'LSTM':
        embedding, embedding_dim = create_emb_layer(make_weight_matrix(words))
        model = LSTM(embedding=embedding, emb_dim=embedding_dim, dimension=256, num_layers=2)
    elif model_type == 'BERT':
        # model = AutoModelForTokenClassification.from_pretrained("lanwuwei/BERTOverflow_stackoverflow_github")
        model = BERTClassifier(hidden_dim=256, dropout=0.5)
    # TODO: Should be validation dl
    train_model(model, train_dl, test_dl, test_dl, epochs=epochs, lr=learning_rate, criterion=criterion)
    """
    Best so far:
    -> max length = 32
        python3 train.py --seed 12345 --loss WCE --epochs 10 --batch_size 64 --lr 2e-5 --model BERT
    -> max length = 64, hidden=256
        python3 train.py --seed 12345 --loss WCE --epochs 8 --batch_size 64 --lr 2e-5 --model BERT
    """

if __name__ == '__main__':
    run()
