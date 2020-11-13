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

from tqdm import tqdm

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):

    def __init__(self, embedding, emb_dim=320, dimension=128, num_layers=1):
        super(LSTM, self).__init__()

        self.embedding = embedding
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=dimension,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2 * dimension, 5)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)  # .cuda()  # throw in raw text

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
        return torch.from_numpy(self.X.iloc[idx][0]), self.y.iloc[idx], self.X.iloc[idx][1]


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