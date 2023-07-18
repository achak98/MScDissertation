import os
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from TorchCRF import CRF 
import torch.optim as optim
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser('EDU segmentation toolkit 1.0')
    parser.add_argument('--prepare', action='store_true',
                        help='preprocess the RST-DT data and create the vocabulary')
    parser.add_argument('--train', action='store_true', help='train the segmentation model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--segment', action='store_true', help='segment new files or input text')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float,
                                default=0.001, help='learning rate')
    train_settings.add_argument('--weight_decay', type=float,
                                default=1e-4, help='weight decay')
    train_settings.add_argument('--batch_size', type=int,
                                default=2, help='batch size')
    train_settings.add_argument('--epochs', type=int,
                                default=50, help='train epochs')
    train_settings.add_argument('--seed', type=int,
                                default=42, help='the random seed')
    train_settings.add_argument('--tagset_size', type=int,
                                default=4, help='number of tags in the tagset')
    train_settings.add_argument('--hidden_size', type=int,
                                default=256, help='hidden size')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--rst_dir', default='../Data/rst/',
                               help='the path of the rst data directory')
    path_settings.add_argument('--seg_data_path',
                               help='the path of the data to segment')
    path_settings.add_argument('--model_dir', default='../Data/models/',
                               help='the dir to save the model')
    path_settings.add_argument('--result_dir', default='../Data/results',
                               help='the directory to save edu segmentation results')
    path_settings.add_argument('--log_path', help='the file to output log')
    return parser.parse_args()

def preprocess_RST_Discourse_dataset(path_data, tag2idx):
    """
    This function preprocesses the RST Discourse dataset.
    """
    text_files = sorted([f for f in os.listdir(path_data) if f.endswith('.out')])
    edus_files = sorted([f for f in os.listdir(path_data) if f.endswith('.edus')])

    data = []
    for txt_file, edu_file in zip(text_files, edus_files):
        with open(os.path.join(path_data, txt_file), 'r') as txtf, open(os.path.join(path_data, edu_file), 'r') as eduf:
            text = txtf.read()
            edus = eduf.read().split('\n')

            # Assume that EDUs are separated by two spaces in the original text
            words = text.split(' ')
            BIOE_tags = []
            edu_idx = 0
            for word in words:
                if word == '':
                    continue
                if word in edus[edu_idx]:
                    if edus[edu_idx].startswith(word):
                        BIOE_tags.append(tag2idx['B'])
                    elif edus[edu_idx] == word:
                        BIOE_tags.append(tag2idx['I'])
                    elif edus[edu_idx].endswith(word):
                        BIOE_tags.append(tag2idx['E'])
                    edu_idx = min(edu_idx + 1, len(edus) - 1)
                else:
                    BIOE_tags.append(tag2idx['O'])

            data.append((words, BIOE_tags))

    df = pd.DataFrame(data, columns=['Text', 'BIOE'])
    return df

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = nn.functional.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class EDUPredictor(nn.Module):
    def __init__(self, tagset_size=4, hidden_dim=2046):
        super(EDUPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.transformer_architecture = 'microsoft/deberta-v3-base'
        self.config = AutoConfig.from_pretrained(self.transformer_architecture, output_hidden_states=True)
        self.config.max_position_embeddings = hidden_dim
        self.encoder = AutoModel.from_pretrained(self.transformer_architecture, config=self.config)
        self.tokeniser = AutoTokenizer.from_pretrained(self.transformer_architecture, max_length=self.config.max_position_embeddings, padding="max_length", return_attention_mask=True)
        # Define BiLSTM 1
        self.lstm1 = nn.LSTM(self.encoder.config.hidden_size, hidden_dim // 2, bidirectional=True)

        # Define self-attention
        self.self_attention = SelfAttention(hidden_dim)

        # Define BiLSTM 2
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True)

        # Define MLP
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Define CRF
        self.crf = CRF(tagset_size)

    def forward(self, sentences, attn_masks):
        encoded_layers = self.encoder(sentences, attention_mask=attn_masks)
        hidden_states = encoded_layers.last_hidden_state
        lstm_out, _ = self.lstm1(hidden_states)
        attn_out, attention_weights = self.self_attention(lstm_out)
        lstm_out, _ = self.lstm2(attn_out.unsqueeze(1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentences), -1))
        tag_scores = self.crf.decode(tag_space)

        return tag_scores, attention_weights

def main():
    args = parse_args()

    # Define the mapping from index to tag
    idx2tag = {0: 'B', 1: 'I', 2: 'O', 3: 'E'}
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}

    if args.prepare:
        # Use the function
        train_df = preprocess_RST_Discourse_dataset(os.path.join(args.rst_dir, "train"), tag2idx)
        # Save the dataframe to a csv file for further use
        train_df.to_csv(os.path.join(args.rst_dir, 'preprocessed_data_train.csv'), index=False)
        test_df = preprocess_RST_Discourse_dataset(os.path.join(args.rst_dir, "test"), tag2idx)
        # Save the dataframe to a csv file for further use
        test_df.to_csv(os.path.join(args.rst_dir, 'preprocessed_data_test.csv'), index=False)

    # Detect device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = EDUPredictor(tagset_size=len(idx2tag.keys()), hidden_dim=args.hidden_size).to(device)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.train:
         # Convert data to PyTorch tensors and move to the device
        train_data = pd.read_csv(os.path.join(args.rst_dir, 'preprocessed_data_train.csv'))
        tokenised_inputs = model.tokeniser(train_data['Text'].tolist(), return_attention_mask=True)
        train_inputs = tokenised_inputs["input_ids"]
        attention_masks = tokenised_inputs["attention_mask"]
        train_tuples = zip(train_inputs, attention_masks)
        train_labels = torch.tensor(train_data['BIOE'].tolist(), dtype=torch.long).to(device)

        # Create DataLoader for training data
        train_dataset = torch.utils.data.TensorDataset(train_tuples, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # Training loop
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            model.train()  # Set model to training mode
            for (inputs, attention_masks), labels in train_loader:
                inputs = inputs.to(device)
                labels = [[tag for tag in row] for row in labels]
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                optimizer.zero_grad()  # Zero the gradients

                # Forward propagation
                tag_scores, _ = model(inputs, attention_masks)

                # Compute the loss
                loss = -model.crf(tag_scores, labels)

                # Backward propagation
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimization step
                optimizer.step()

                epoch_loss += loss.item()

            # Print epoch statistics
            print(f'Epoch {epoch+1} Loss: {epoch_loss:.3f}')

        # Save the trained model
        model_path = os.path.join(args.model_dir, 'edu_segmentation_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to: {model_path}")

    if args.evaluate:
        test_data = pd.read_csv(os.path.join(args.rst_dir, 'preprocessed_data_test.csv'))
        test_inputs = torch.tensor(test_data['Text'], dtype=torch.long).to(device)
        test_labels = torch.tensor(test_data['BIOE'], dtype=torch.long).to(device)
        # Load the trained model
        model_path = os.path.join(args.model_dir, 'edu_segmentation_model.pt')
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from: {model_path}")

        # Evaluation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Predict output for test set
            test_tag_scores, _ = model(test_inputs)
            test_pred = model.crf.decode(test_tag_scores)

            # Flatten both labels and predictions
            test_tags = [idx2tag[i] for row in test_labels for i in row]
            test_pred_tags = [idx2tag[i] for row in test_pred for i in row]

            # Compute evaluation metrics
            accuracy = accuracy_score(test_tags, test_pred_tags)
            precision = precision_score(test_tags, test_pred_tags)
            recall = recall_score(test_tags, test_pred_tags)
            f1 = f1_score(test_tags, test_pred_tags)

            print(f'Test Accuracy: {accuracy:.3f}')
            print(f'Test Precision: {precision:.3f}')
            print(f'Test Recall: {recall:.3f}')
            print(f'Test F1-Score: {f1:.3f}')

            with open(os.path.join(args.results_dir, "edu_results.txt"), 'w') as file:
                file.write(f'Test Accuracy: {accuracy:.3f}\nTest Precision: {precision:.3f}\nTest Recall: {recall:.3f}\nTest F1-Score: {f1:.3f}')

            # Generate confusion matrix
            confusion_matrix = multilabel_confusion_matrix(test_tags, test_pred_tags)
            for i, matrix in enumerate(confusion_matrix):
                plt.figure()
                sns.heatmap(matrix, annot=True, fmt='d')
                plt.title(f'Confusion Matrix for tag {i}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(args.results_dir, 'confusion_matrix_edu.pdf'))

    if args.segment:
        text_files = sorted([f for f in os.listdir(args.seg_data_path) if f.endswith('.txt')]) #TODO: FIGURE THIS OUT
        data = []
        for txt_file in text_files:
            with open(os.path.join(args.seg_data_path, txt_file), 'r') as txtf:
                text = txtf.read()
                # Assume that EDUs are separated by two spaces in the original text
                words = text.split(' ')
                data.append(words)
        seg_ip = pd.DataFrame(data)
        model.eval()
        with torch.no_grad():
            # Predict output for test set
            seg_tag_scores, _ = model(seg_ip)
            seg_pred = model.crf.decode(seg_tag_scores)

        #TODO: FIGURE OUT HOW TO SAVE THE DATA
        seg_df = pd.DataFrame(data, columns=['Text', 'BIOE'])
        seg_df.to_csv('preprocessed_data_test.csv', index=False)
if __name__ == '__main__':
    main()
