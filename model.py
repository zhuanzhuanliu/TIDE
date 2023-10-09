from __future__ import absolute_import, division, print_function, unicode_literals
from tkinter import W


import mindspore
import numpy as np



class BiLSTMAttn(mindspore.nn.Cell):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = mindspore.nn.Dropout(dropout)
        self.encoder = mindspore.nn.LSTM(embedding_dim, 
                               hidden_dim // 2, 
                               dropout=dropout if num_layers > 1 else 0,
                               num_layers=num_layers, 
                               batch_first=True, 
                               bidirectional=True)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = mindspore.ops.squeeze(final_hidden, axis=0)
        attn_weights = mindspore.ops.bmm(encoder_out, mindspore.ops.unsqueeze(hidden, 2))
        attn_weights = mindspore.ops.squeeze(attn_weights, axis=2)
        soft_attn_weights = mindspore.ops.softmax(attn_weights, 1)
        new_hidden = mindspore.ops.bmm(mindspore.ops.swapaxes(encoder_out, 1, 2), mindspore.ops.unsqueeze(soft_attn_weights, 2))
        new_hidden = mindspore.ops.squeeze(new_hidden, axis=2)

        return new_hidden

    def construct(self, features, lens):
        features = self.dropout(features)
        outputs, (hn, cn) = self.encoder(features,lens)
        fbout = outputs[:, :, :self.hidden_dim // 2] + outputs[:, :, self.hidden_dim // 2:]
        fbhn = (hn[-2, :, :] + hn[-1, :, :])
        fbhn = mindspore.ops.unsqueeze(fbhn, 0)
        attn_out = self.attnetwork(fbout, fbhn)

        return attn_out


class BiLSTM(mindspore.nn.Cell):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = mindspore.nn.Dropout(dropout)
        self.bilstm = mindspore.nn.LSTM(embedding_dim, 
                              hidden_dim // 2, 
                              dropout=dropout, 
                              num_layers=num_layers, 
                              batch_first=True,
                              bidirectional=True)

    def construct(self, features, lens):
        features = self.dropout(features)
        outputs, hidden_state = self.bilstm(features)

        return outputs, hidden_state  # outputs: batch, seq, hidden_dim - hidden_state: hn, cn: 2*num_layer, batch_size, hidden_dim/2


class HistoricCurrent(mindspore.nn.Cell):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, model, class_num, device):
        super().__init__()
        self.model = model
        if self.model == "tlstm":
            self.historic_model = TimeLSTM(embedding_dim, hidden_dim, device)
        elif self.model == "bilstm":
            self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        elif self.model == "bilstm-attention":
            self.historic_model = BiLSTMAttn(embedding_dim, hidden_dim, num_layers, dropout)

        self.fc_ct = mindspore.nn.Dense(embedding_dim, hidden_dim)
        self.fc_ct_attn = mindspore.nn.Dense(embedding_dim, hidden_dim//2)

        self.fc_concat = mindspore.nn.Dense(hidden_dim * 2, hidden_dim)
        self.fc_concat_attn = mindspore.nn.Dense(hidden_dim, hidden_dim)

        self.dropout = mindspore.nn.Dropout(dropout)
        self.final = mindspore.nn.Dense(hidden_dim, class_num)

    @staticmethod
    def combine_features(tweet_features, historic_features):
        return mindspore.ops.cat((tweet_features, historic_features), 1)

    def construct(self, tweet_features, historic_features, lens, timestamp):
        if self.model == "tlstm":
            outputs = self.historic_model(historic_features, timestamp)
            tweet_features = mindspore.ops.relu(self.fc_ct(tweet_features.astype('float32')))
            outputs = mindspore.ops.mean(outputs, 1)
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = mindspore.ops.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm":
            
            historic_features = historic_features[:,mindspore.ops.randperm(1),:]

            outputs, (h_n, c_n) = self.historic_model(historic_features, lens)
            outputs = mindspore.ops.mean(outputs, 1)
            tweet_features = mindspore.ops.relu(self.fc_ct(tweet_features.astype('float32')))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = mindspore.ops.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm-attention":
            outputs = self.historic_model(historic_features, lens)
            tweet_features = mindspore.ops.relu(self.fc_ct_attn(tweet_features))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = mindspore.ops.relu(self.fc_concat_attn(combined_features))

        x = self.dropout(x)

        return self.final(x)


class Historic(mindspore.nn.Cell):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, class_num):
        super().__init__()
        self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        self.dropout = mindspore.nn.Dropout(dropout)
        self.fc1 = mindspore.nn.Dense(hidden_dim, 32)
        self.final = mindspore.nn.Dense(32, class_num)

    def construct(self, tweet_features, historic_features, lens, timestamp):
        outputs, (h_n, c_n) = self.historic_model(historic_features, lens)
        hidden = mindspore.ops.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = mindspore.ops.relu(self.fc1(hidden))
        return self.final(x)


class Current(mindspore.nn.Cell):
    def __init__(self, hidden_dim, dropout, class_num):
        super().__init__()
        self.fc1 = mindspore.nn.Dense(768, hidden_dim)
        self.dropout = 	mindspore.nn.Dropout(dropout)
        self.fc2 = mindspore.nn.Dense(hidden_dim, 32)
        self.final = mindspore.nn.Dense(32, class_num)

    def construct(self, tweet_features, historic_features, lens, timestamp):
        x = mindspore.ops.relu(self.fc1(tweet_features))
        x = self.dropout(x)
        x = mindspore.ops.relu(self.fc2(x))
        return self.final(x)


class TimeLSTM(mindspore.nn.Cell):
    def __init__(self, input_size, hidden_size, device,bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = mindspore.nn.Dense(hidden_size, hidden_size * 4)
        self.U_all = mindspore.nn.Dense(input_size, hidden_size * 4)
        self.W_d = mindspore.nn.Dense(hidden_size, hidden_size)
        self.bidirectional = bidirectional
        self.device = device


    def construct(self, inputs, timestamps, reverse=False):
        b, seq, embed = mindspore.ops.shape(inputs)
        h = mindspore.ops.zeros((b, self.hidden_size))
        c = mindspore.ops.zeros((b, self.hidden_size))

        outputs = []
        for s in range(seq):
            c_s1 = mindspore.ops.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = mindspore.ops.chunk(outs, 4, 1)
            f = mindspore.ops.sigmoid(f)
            i = mindspore.ops.sigmoid(i)
            o = mindspore.ops.sigmoid(o)
            c_tmp = mindspore.ops.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * mindspore.ops.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = mindspore.ops.stack(outputs, 1)
        return outputs
