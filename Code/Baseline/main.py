import torch
import torch.nn as nn
import torch.optim as optim
import argsparser, dataloaders, evaluation, train
import gensim.downloader as api

# Parse the arguments
args = argsparser.parse_args()

# Access the parsed arguments
dataDir = args.dataDir
vocab_size = args.vocab_size
embedding_dim = args.embedding_dim
num_epochs = args.num_epochs
lr = args.lr
cnn_window_size_small = args.cnn_window_size_small
cnn_window_size_medium = args.cnn_window_size_medium
cnn_window_size_large = args.cnn_window_size_large
batch_size = args.batch_size
bgru_hidden_size = args.bgru_hidden_size
dropout = args.dropout
mode = args.mode
log_interval = args.log_interval
numOfWorkers = args.numOfWorkers
cnnfilters = args.cnnfilters

w2v_model = api.load("word2vec-google-news-300")

# Define your model architecture
class Baseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.conv1 = nn.Conv2d(embedding_dim, cnnfilters, cnn_window_size_small)
        self.pool1 = nn.MaxPool2d(cnn_window_size_small)
        self.gru1 = nn.GRU(cnnfilters, bgru_hidden_size, batch_first=True, bidirectional=True)
        
        self.conv2 = nn.Conv2d(embedding_dim, cnnfilters, cnn_window_size_medium)
        self.pool2 = nn.MaxPool2d(cnn_window_size_medium)
        self.gru2 = nn.GRU(cnnfilters, bgru_hidden_size, batch_first=True, bidirectional=True)
        
        self.conv3 = nn.Conv2d(embedding_dim, cnnfilters, cnn_window_size_large)
        self.pool3 = nn.MaxPool2d(cnn_window_size_large)
        self.gru3 = nn.GRU(cnnfilters, bgru_hidden_size, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(bgru_hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)

        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        _, h1 = self.gru1(x1)
        h1 = torch.cat((h1[-2, :, :], h1[-1, :, :]), dim=1)

        x2 = self.conv1(x)
        x2 = self.pool1(x2)
        _, h2 = self.gru1(x2)
        h2 = torch.cat((h2[-2, :, :], h2[-1, :, :]), dim=1)

        x3 = self.conv1(x)
        x3 = self.pool1(x3)
        _, h3 = self.gru1(x3)
        h3 = torch.cat((h3[-2, :, :], h3[-1, :, :]), dim=1)

        h = torch.cat((h1, h2, h3), dim=-1)

        h = self.fc(h)
        h = self.sigmoid(h)

def main():
    # Instantiate your model
    model = Baseline(vocab_size=vocab_size, embedding_dim=embedding_dim)

    # Define your loss function and optimizer
    train_dataloader, val_dataloader, test_dataloader = dataloaders.create_data_loaders(dataDir, batch_size)
    train.train(model, train_dataloader, val_dataloader, num_epochs, lr)
    qwk_score = evaluation.evaluate(model, test_dataloader)
    print("Quadratic Weighted Kappa (QWK) Score on test set:", qwk_score)

if __name__ == "__main__":
    main()