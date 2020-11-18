import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

USE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):

    def __init__(self, embedding, emb_dim=320, dimension=128, num_layers=1):
        super(LSTMClassifier, self).__init__()

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
        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # better to pack the text

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_features = self.drop(out_reduced)

        text_features = self.fc(text_features)

        return text_features
