import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.6):
        super(BERTClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        configuration = BertConfig()
        self.bert_layer_size = configuration.hidden_size
        # value = bert-base-uncased
        # lanwuwei/BERTOverflow_stackoverflow_github
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dense_0 = nn.Linear(self.bert_layer_size, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.dense_1 = nn.Linear(self.hidden_dim, 5)

    def forward(self, inputs, length):
        """
        length argument not used in BERT
        """
        input_ids, token_type_ids, attn_mask = inputs.permute(1, 0, 2)
        _, pooled_output, attn_states = self.bert(input_ids, token_type_ids, attn_mask)
        # dense 0 + relu
        output = self.relu(self.dense_0(pooled_output))
        # dropout
        output = self.dropout_layer(output)
        # dense 1
        output = self.dense_1(output)
        return output
