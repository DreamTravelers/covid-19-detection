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
        self.fn_lstm = nn.Linear(128, 3, bias=False)

    def forward(self, x):
        input = self.embedding(x.squeeze())
        lstm_output, _ = self.lstm(input)
        output = self.avgpool(lstm_output)
        output = output.squeeze()
        forward_lstm, backward_lstm = lstm_output[:, :, :128], lstm_output[:, :, 128:]
        l_t = forward_lstm + backward_lstm
        output = self.fn_lstm(l_t[:, -1, :])
        return output, l_t


class fuse_cv_text_model(nn.Module):
    def __init__(self, cv_model, text_model) -> None:
        super(fuse_cv_text_model, self).__init__()
        self.vgg_model = cv_model
        self.text_model = text_model

        self.fc_model = nn.Linear(128, 3)

        self.cvdrop = nn.Dropout(0.2)
        self.textdrop = nn.Dropout(0.3)

        self.cv_qv = nn.Linear(128, 128 * 2)
        self.text_qv = nn.Linear(128, 128 * 2)

        self.mlp = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, cv_input, text_input):
        cv_dim = self.vgg_model(cv_input)
        avg_lstm_output, last_lstm_output = self.text_model(text_input)

        cv_q, cv_v = self.cv_qv(cv_dim).chunk(2, dim=-1)
        text_q, text_v = self.text_qv(avg_lstm_output).chunk(2, dim=-1)

        cv_x = self.tanh(cv_q) * self.tanh(text_q)
        # cv_hm = self.softmax(cv_x)
        cv_hm = cv_x / abs(cv_x.min())
        text_x = self.tanh(text_v) * self.tanh(cv_v)
        # text_hm = self.softmax(text_x)
        text_hm = text_x / abs(text_x.min())

        output1 = cv_hm + text_hm
        output = self.mlp(output1)
        return output

        # cv_q,cv_v = self.cv_qv(cv_dim).chunk(2,dim=-1)
        # text_q,text_v = self.text_qv(avg_lstm_output).chunk(2,dim=-1)
        # cv_q = self.cvdrop(cv_dim)
        # cv_v = cv_dim

        # text_q = self.textdrop(avg_lstm_output)
        # text_v =avg_lstm_output

        # # image-based text attention
        # cv_dots = torch.matmul(cv_q,avg_lstm_output.transpose(1,0)) * (1/math.sqrt(cv_dim.size(-1)))
        # cv_attention = torch.matmul(self.softmax(cv_dots*10000),cv_v)

        # # Text_based image attention
        # text_dots =torch.matmul(text_q,cv_dim.transpose(1,0)) * (1/math.sqrt(cv_dim.size(-1)))
        # text_attention = torch.matmul(self.softmax(text_dots*10000),text_v)

        # cv_text = cv_attention + text_attention
        # output = self.mlp(cv_text)

        # output1 = cv_dim + avg_lstm_output
        # output = self.mlp(output1)
        # return output
