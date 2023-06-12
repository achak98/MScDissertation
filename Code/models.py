import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embedded = self.embeddings(target)
        output = self.linear(embedded)
        return output

class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        self.conv1 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_small)
        self.pool1 = nn.MaxPool1d(kernel_size=args.cnn_window_size_small, stride=1)
        self.gru1 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_medium)
        self.pool2 = nn.MaxPool1d(kernel_size=args.cnn_window_size_small, stride=1)
        self.gru2 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=args.dropout)

        self.conv3 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_large)
        self.pool3 = nn.MaxPool1d(kernel_size=args.cnn_window_size_small, stride=1)
        self.gru3 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(p=args.dropout)

        self.fc = nn.Linear(args.bgru_hidden_size*2*3,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=x.permute(0,2,1)
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        x1=x1.permute(0,2,1)
        _, h1 = self.gru1(x1) #x1 should be batch size, sequence length, input length
        h1 = self.dropout1(h1)
        h1 = torch.cat((h1[0, :, :], h1[1, :, :]), dim=1)

        x2 = self.conv2(x)
        x2 = self.pool2(x2)
        x2=x2.permute(0,2,1)
        _, h2 = self.gru2(x2)
        h2 = self.dropout1(h2)
        h2 = torch.cat((h2[0, :, :], h2[1, :, :]), dim=1)

        x3 = self.conv3(x)
        x3 = self.pool3(x3)
        x3=x3.permute(0,2,1)
        _, h3 = self.gru3(x3)
        h3 = self.dropout1(h3)
        h3 = torch.cat((h3[0, :, :], h3[1, :, :]), dim=1)

        h = torch.cat((h1, h2, h3), dim=1)
        h = self.fc(h)
        h = self.sigmoid(h)

        return h.squeeze()
