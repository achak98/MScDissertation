    --embedding_dim 300 \
    --num_epochs 40 \
    --batch_size 32 \
    --lr 0.001 \
    --mode train \
    --cnnfilters 100 \
    --cnn_window_size_small 2 \
    --cnn_window_size_medium 3 \
    --cnn_window_size_large 4 \
    --bgru_hidden_size 128 \
    --dropout 0.4 \
    --log_interval 10 \

class Baseline(nn.Module):
    def __init__(self, args, strides, kernels):
        super(Baseline, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        self.conv1 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_small, stride = strides[0])
        self.pool1 = nn.MaxPool1d(kernel_size=kernels[0], stride=strides[1])
        self.gru1 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_medium, stride=strides[2])
        self.pool2 = nn.MaxPool1d(kernel_size=kernels[1], stride=strides[3])
        self.gru2 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=args.dropout)

        self.conv3 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_large, stride=strides[4])
        self.pool3 = nn.MaxPool1d(kernel_size=kernels[2], stride=strides[5])
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
        h1 = torch.cat((h1[0, :, :], h1[1, :, :]), dim=1)
        h1 = self.dropout1(h1)

        x2 = self.conv2(x)
        x2 = self.pool2(x2)
        x2=x2.permute(0,2,1)
        _, h2 = self.gru2(x2)
        h2 = torch.cat((h2[0, :, :], h2[1, :, :]), dim=1)
        h2 = self.dropout1(h2)

        x3 = self.conv3(x)
        x3 = self.pool3(x3)
        x3=x3.permute(0,2,1)
        _, h3 = self.gru3(x3)
        h3 = torch.cat((h3[0, :, :], h3[1, :, :]), dim=1)
        h3 = self.dropout1(h3)

        h = torch.cat((h1, h2, h3), dim=1)
        h = self.fc(h)
        h = self.sigmoid(h)

        return h.squeeze()