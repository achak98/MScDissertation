import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--dataDir', type=str, help='Data Directory')
    parser.add_argument('--skipgram_file_path', type=str, help='Skipgram File Directory')
    parser.add_argument('--numOfWorkers', type=str, help='Number of Workers for Dataloaders')
    parser.add_argument('--embedding_dim', type=int, default = 300, help='Embedding Dimension')
    parser.add_argument('--num_epochs', type=int, default = 40, help='No. of Epochs')
    parser.add_argument('--batch_size', type= int, default = 128, help = 'Batch Size')
    parser.add_argument('--lr', type=float, default = 0.001, help='Learning Rate')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode (train or test)')
    parser.add_argument('--cnnfilters', type= int, default = 100, help = 'CNN Filters')
    parser.add_argument('--cnn_window_size_small', type= int, default = 2, help = 'Smallest CNN Window Size')
    parser.add_argument('--cnn_window_size_medium', type= int, default = 3, help = 'Medium CNN Window Size')
    parser.add_argument('--cnn_window_size_large', type= int, default = 4, help = 'largest CNN Window Size')
    parser.add_argument('--bgru_hidden_size', type= int, default = 128, help = 'No. of Bi-GRU hidden units')
    parser.add_argument('--prompt', type= int, default = 1, help = 'Prompt to train and test for')
    parser.add_argument('--max_length', type= int, default = 768, help = 'Padding sequence length')
    parser.add_argument('--dropout', type= float, default = 0.4, help = 'Dropout Rate')
    parser.add_argument('--log_interval', type= int, default = 0.4, help = 'Loggin Interval in Epochs')
    return parser.parse_args()