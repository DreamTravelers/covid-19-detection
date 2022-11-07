import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextLstm(nn.Module):
    def __init__(self) -> None:
        super(TextLstm, self).__init__()

        self.embedding = nn.Embedding(119547, 256, padding_idx=0)
        self.lstm = nn.LSTM(256, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.avgpool = nn.AvgPool2d((40, 1), stride=2)

    def forward(self, x):
        input = self.embedding(x.squeeze())
        lstm_output, _ = self.lstm(input)
        output = self.avgpool(lstm_output)
        output = output.squeeze()
        return output, lstm_output[:, -1, :]


class fuse_cv_text_model(nn.Module):
    def __init__(self, cv_model, text_model) -> None:
        super(fuse_cv_text_model, self).__init__()
        self.vgg_model = cv_model
        self.text_model = text_model

        self.fc_model = nn.Linear(128, 3)

        self.cv_qv = nn.Linear(128, 128 * 2)
        self.text_qv = nn.Linear(128, 128 * 2)

        self.mlp = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cv_input, text_input):
        cv_dim = self.vgg_model(cv_input)
        avg_lstm_output, last_lstm_output = self.text_model(text_input)

        cv_q, cv_v = self.cv_qv(cv_dim).chunk(2, dim=-1)
        text_q, text_v = self.text_qv(avg_lstm_output).chunk(2, dim=-1)

        # image-based text attention
        cv_dots = torch.matmul(cv_q, avg_lstm_output.transpose(1, 0)) * (1 / math.sqrt(cv_dim.size(-1)))
        cv_attention = torch.matmul(self.softmax(cv_dots), cv_v)

        # Text_based image attention
        text_dots = torch.matmul(text_q, cv_dim.transpose(1, 0)) * (1 / math.sqrt(cv_dim.size(-1)))
        text_attention = torch.matmul(self.softmax(text_dots), text_v)

        cv_text = cv_attention + text_attention

        output = self.mlp(cv_text)
        return self.softmax(output)
