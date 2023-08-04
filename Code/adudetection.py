import os
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaV3Model

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
    path_settings.add_argument('--pe_dir', default='../Data/pe/',
                               help='the path of the PE data directory')
    path_settings.add_argument('--train_test_split_file', default = "train-test-split.csv", 
                               help = 'split file name')
    path_settings.add_argument('--prompts_file', default = "prompts.csv", 
                               help = 'prompts file name')
    path_settings.add_argument('--output_train_file', default = "preprocessed_data_adu_train.csv", 
                               help = 'preprocessed_train file name')
    path_settings.add_argument('--output_test_file', default = "preprocessed_data_adu_test.csv", 
                               help = 'preprocessed_test file name')
    path_settings.add_argument('--seg_data_path',
                               help='the path of the data to segment')
    path_settings.add_argument('--model_dir', default='../Data/models/',
                               help='the dir to save the model')
    path_settings.add_argument('--result_dir', default='../Data/results',
                               help='the directory to save adu segmentation results')
    path_settings.add_argument('--log_path', help='the file to output log')
    return parser.parse_args()

import csv

def preprocess_PE_dataset(input_folder, train_test_split_file, prompts_file, output_train_file, output_test_file, tag2idx):
    # Read train-test split file
    train_test_split = {}
    with open(train_test_split_file, 'r') as split_file:
        reader = csv.reader(split_file)
        next(reader)  # Skip header row
        for row in reader:
            file_name, split = row
            train_test_split[file_name] = split

    # Read prompts file
    prompts = {}
    with open(prompts_file, 'r') as prompts_csv:
        reader = csv.reader(prompts_csv)
        next(reader)  # Skip header row
        for row in reader:
            file_name, prompt = row
            prompts[file_name] = prompt

# Process each file
    with open(output_train_file, 'w', newline='') as train_csv, open(output_test_file, 'w', newline='') as test_csv:
        train_writer = csv.writer(train_csv)
        test_writer = csv.writer(test_csv)

        # Write header row
        train_writer.writerow(["Essay ID", "Prompt", "Text", "ADU"])
        test_writer.writerow(["Essay ID", "Prompt", "Text", "ADU"])

        # Process each file in the input folder
        for file_name in train_test_split.keys():
            txt_file_path = f"{input_folder}/{file_name}.txt"
            ann_file_path = f"{input_folder}/{file_name}.ann"

            # Read the text content
            with open(txt_file_path, 'r') as txt_file:
                text = txt_file.read().replace('\n', ' ')

            # Read the annotation file
            adus = []
            with open(ann_file_path, 'r') as ann_file:
                lines = ann_file.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("T"):
                        adu_label, adu_start, adu_end = line.split("\t")[1].split(" ")
                        adus.append((int(adu_start), int(adu_end), adu_label))

            # Sort ADUs based on their start position
            adus.sort(key=lambda x: x[0])

            # Determine if the file belongs to train or test set
            split = train_test_split[file_name]
            writer = train_writer if split == "train" else test_writer

            # Generate BIOE labels for the ADUs
            adu_labels = [tag2idx["O"]] * len(text)
            for start, end, label in adus:
                for i in range(start, end+1):
                    adu_labels[i] = tag2idx["I"]

            # Write the preprocessed data to the appropriate file
            writer.writerow([file_name, prompts[file_name], text, adu_labels])



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

class ADUPredictor(nn.Module):
    def __init__(self, tagset_size=1, hidden_dim=512):
        super(ADUPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.transformer_architecture = 'allenai/longformer-base-4096'
        self.config = AutoConfig.from_pretrained(self.transformer_architecture, output_hidden_states=True)
        self.config.max_position_embeddings = 2048
        self.encoder = AutoModel.from_pretrained(self.transformer_architecture, config=self.config)
        # Define BiLSTM 1
        self.lstm1 = nn.LSTM(self.encoder.config.hidden_size, hidden_dim // 2, bidirectional=True)

        # Define self-attention
        self.self_attention = SelfAttention(hidden_dim)

        # Define BiLSTM 2
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True)

        # Define MLP
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        #self.softmax = nn.Softmax(dim=1)
        # Define CRF
        #self.crf = CRF(tagset_size)

    def forward(self, sentences):
        encoded_layers = self.encoder(sentences)
        hidden_states = encoded_layers.last_hidden_state
        lstm_out, _ = self.lstm1(hidden_states)
        attn_out, attention_weights = self.self_attention(lstm_out)
        lstm_out, _ = self.lstm2(attn_out.unsqueeze(1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentences), -1))

        return tag_space

def main():
    args = parse_args()

    # Define the mapping from index to tag
    #idx2tag = {0: 'B-MajorClaim', 1: 'B-Claim', 2: 'B-Premise', 3: 'I-MajorClaim', 4: 'I-Claim', 5: 'I-Premise', 6:'O', 7: 'E-MajorClaim', 8: 'E-Claim', 9: 'E-Premise'}
    idx2tag = {0:"I", 1:"O"}
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}
    

    if args.prepare:
        # Use the functiontrain-test-split
        preprocess_PE_dataset(os.path.join(args.pe_dir), args.train_test_split_file, args.prompts_file, args.output_train_file, args.output_test_file, tag2idx) 
       

    # Detect device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = ADUPredictor(tagset_size=len(idx2tag.keys()), hidden_dim=args.hidden_size).to(device)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    if args.train:
         # Convert data to PyTorch tensors and move to the device
        train_data = pd.read_csv(os.path.join(args.pe_dir, 'preprocessed_data_adu_train.csv'))
        "Prompt", "Text", "ADU"
        train_inputs = torch.tensor(train_data['Text'], dtype=torch.long).to(device)
        train_labels = torch.tensor(train_data['BIOE'], dtype=torch.long).to(device)

        # Create DataLoader for training data
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # Training loop
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            model.train()  # Set model to training mode
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = [[tag for tag in row] for row in labels]
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                optimizer.zero_grad()  # Zero the gradients

                # Forward propagation
                tag_space = model(inputs)

                # Compute the loss
                loss = criterion(tag_space, labels)
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
        model_path = os.path.join(args.model_dir, 'adu_segmentation_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to: {model_path}")

    if args.evaluate:
        test_data = pd.read_csv(os.path.join(args.pe_dir, 'preprocessed_data_adu_test.csv'))
        test_inputs = torch.tensor(test_data['Text'], dtype=torch.long).to(device)
        test_labels = torch.tensor(test_data['BIOE'], dtype=torch.long).to(device)
        # Load the trained model
        model_path = os.path.join(args.model_dir, 'adu_segmentation_model.pt')
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from: {model_path}")

        # Evaluation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Predict output for test set
            test_tag_scores = model(test_inputs)
            sig = nn.Sigmoid()
            test_pred = (sig(test_tag_scores) > 0.5).float()

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
            with open(os.path.join(args.results_dir, "adu_results.txt"), 'w') as file:
                file.write(f'Test Accuracy: {accuracy:.3f}\nTest Precision: {precision:.3f}\nTest Recall: {recall:.3f}\nTest F1-Score: {f1:.3f}')

            # Generate confusion matrix
            confusion_matrix = multilabel_confusion_matrix(test_tags, test_pred_tags)
            for i, matrix in enumerate(confusion_matrix):
                plt.figure()
                sns.heatmap(matrix, annot=True, fmt='d')
                plt.title(f'Confusion Matrix for tag {i}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(args.results_dir, 'confusion_matrix_adu.pdf'))

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
        seg_df.to_csv('preprocessed_data_adu_test.csv', index=False)
if __name__ == '__main__':
    main()
