import os
import ast, re
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcrf import CRF 
import torch.optim as optim
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore') 

def compute_f1_score_for_labels(y_true, y_pred, labels):
    # y_true: Ground truth labels (list of lists)
    # y_pred: Predicted labels (list of lists)
    # labels: List of unique labels in your dataset
    # Flatten the ground truth and predicted labels
    #y_true_flat = [label for sublist in y_true for label in sublist]
    #y_pred_flat = [label for sublist in y_pred for label in sublist]

    # Compute precision, recall, and F1 score for each label
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels)

    # Create a dictionary to store the results for each label
    label_scores = {}
    for i, label in enumerate(labels):
        label_scores[label] = {
            'Precision': precision[i],
            'Recall': recall[i],
            'F1 Score': f1_score[i]
        }

    return label_scores

def parse_args():
    parser = argparse.ArgumentParser('EDU segmentation toolkit 1.0')
    parser.add_argument('--prepare', action='store_true',
                        help='preprocess the RST-DT data and create the vocabulary')
    parser.add_argument('--train', action='store_true', help='train the segmentation model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--segment', action='store_true', help='segment new files or input text')

    parser.add_argument('--regex_pattern', default= r'\b\w+\b|[.,:\n&!]')
    parser.add_argument('--max_length', type=int, default= 18432)
    parser.add_argument('--learning_rate', type=float,
                                default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                                default=3e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int,
                                default=2, help='batch size')
    parser.add_argument('--epochs', type=int,
                                default=100, help='train epochs')
    parser.add_argument('--seed', type=int,
                                default=42, help='the random seed')
    parser.add_argument('--tagset_size', type=int,
                                default=4, help='number of tags in the tagset')
    parser.add_argument('--hidden_dim', type=int,
                                default=768, help='hidden size')
    parser.add_argument('--max_grad_norm', type=float,
                                default=1.0, help='gradient norm')
    parser.add_argument('--rst_dir', default='../Data/rst/',
                               help='the path of the rst data directory')
    parser.add_argument('--seg_data_path',
                               help='the path of the data to segment')
    parser.add_argument('--model_dir', default='../Data/models/',
                               help='the dir to save the model')
    parser.add_argument('--result_dir', default='../Data/results',
                               help='the directory to save edu segmentation results')
    parser.add_argument('--log_path', help='the file to output log')
    return parser.parse_args()


def manual_batching(tensor, batch_size):
    num_samples = tensor.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Create a list to store the batches
    batches = []

    for i in range(num_batches):
        # Calculate the start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        # Extract the current batch from the tensor
        batch = tensor[start_idx:end_idx]

        # Append the batch to the list
        batches.append(batch)

    return batches

def find_sequence_spans(text, target_sequences, model, args):
    sequence_spans = []
    idx_edu = 0
    start_index = None
    loop = True
    i = 0
    while (loop):
        #target_stuff = re.findall(regex_pattern, target_sequences[idx_edu])
        #print(f"target_sequences[idx_edu]: {target_sequences[idx_edu]}")
        target_stuff = model.tokeniser(target_sequences[idx_edu])["input_ids"]
        target_stuff = target_stuff[1:-1]
        target_length = len(target_stuff)
        if len(target_stuff) == 0 :
            break
        if text[i] == target_stuff[0]:
            start_index = i
            idx_edu += 1
            potential_end = i + target_length -1
            #if text[potential_end] != target_stuff[-1]:
            #print(f"potential_end: {potential_end} and idx_edu: {idx_edu} and text[potential_end-1]: {text[potential_end-1]} aand text[potential_end]: {text[potential_end]} and target_stuff[-1]: {target_stuff[-1]}")
            #print(f"len(text): {len(text)} and potential_end: {potential_end} and target_length: {target_length} and target_stuff: {target_stuff} and idx_edu: {idx_edu} and len(target_sequences): {len(target_sequences)}")
            if potential_end > len(text)-1 :
                if target_stuff[len(text)-1 - i] == text[-1]:
                    potential_end = len(text)-1 - i
                else:
                    break
            if text[potential_end] == target_stuff[-1]:
                end_index = potential_end
                i = end_index
                sequence_spans.append((start_index, end_index))
            start_index = None
        #else:
        #  print(f"i: {i}, text[i]: {text[i]}, target_stuff[0]:{target_stuff[0]}")
        i+=1
        if(i>=len(text) or idx_edu >= len(target_sequences)):
          loop = False
    return sequence_spans

def preprocess_RST_Discourse_dataset(path_data, tag2idx, args, model):
    """
    This function preprocesses the RST Discourse dataset.
    """
    text_files = sorted([f for f in os.listdir(path_data) if f.endswith('.out')])
    edus_files = sorted([f for f in os.listdir(path_data) if f.endswith('.edus')])

    data = []
    messed_up_ones = []
    for txt_file, edu_file in zip(text_files, edus_files):
        with open(os.path.join(path_data, txt_file), 'r') as txtf, open(os.path.join(path_data, edu_file), 'r') as eduf:
            text = txtf.read()
            edus = eduf.read().split('\n')
            text = text.split('\n')
            text = ' '.join(text)
            text = text.replace("'", " '\' ").replace("\"", " \" ").replace("-", " - ").replace(",", " , ").replace(".", " . ")
            edus = [edu.replace("'", " '\' ").replace("\"", " \" ").replace("-", " - ").replace(",", " , ").replace(".", " . ") for edu in edus]
            edus = [seq.strip() for seq in edus]

            #words = re.findall(args.regex_pattern, ' '.join(text))
            words = model.tokeniser(text, padding="max_length", truncation = True, return_attention_mask=True, max_length = args.max_length)
            input_ids = words["input_ids"]
            attn_mask = words["attention_mask"]
            BIOE_tags = [tag2idx["O"]] * len(input_ids)
            #print(f"txt_file: {txt_file}")
            sequence_spans = find_sequence_spans(input_ids, edus, model, args)

            for span in sequence_spans:
                if(span[0] == -1 or span[1] == -1):
                    continue
                else:
                    BIOE_tags[span[0]] = tag2idx['B']
                    BIOE_tags[span[1]] = tag2idx['E']
                    for i in range(span[0]+1, span[1]):
                        BIOE_tags[i] = tag2idx['I']
            if(len(sequence_spans) != len(edus) - 1):
                print(f"messed up file: {txt_file}, detected: {len(sequence_spans)}, total: {len(edus)}, length: {len(words['input_ids'])}")
                messed_up_ones.append(txt_file)
            #words = words + ['[PAD]'] * (args.max_length - len(words))    #pads to 18432
            #BIOE_tags = BIOE_tags + [2] * (args.max_length - len(BIOE_tags))    #pads to 18432
            data.append((input_ids, attn_mask, BIOE_tags))

    df = pd.DataFrame(data, columns=['Text', 'Attention Mask', 'BIOE'])
    print(messed_up_ones)
    return df

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = nn.functional.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class EDUPredictor(nn.Module):
    def __init__(self, tagset_size=4, hidden_dim=768, max_length = 18432):
        super(EDUPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.transformer_architecture = 'microsoft/deberta-v3-small'
        self.config = AutoConfig.from_pretrained(self.transformer_architecture, output_hidden_states=True)
        self.config.max_position_embeddings = max_length
        self.encoder = AutoModel.from_pretrained(self.transformer_architecture, config=self.config)
        self.tokeniser = AutoTokenizer.from_pretrained(self.transformer_architecture, max_length=self.config.max_position_embeddings, 
                                                       padding="max_length", return_attention_mask=True)
        # Define BiLSTM 1
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, bidirectional=True)

        # Define self-attention
        #self.self_attention = SelfAttention(hidden_dim)

        # Define BiLSTM 2
        #self.lstm2 = nn.LSTM(hidden_dim//4, hidden_dim // 2, bidirectional=True)

        # Define MLP
        self.hidden2tag = self.regressor1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim//16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim//16, hidden_dim//64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim//64, tagset_size)
        )

        # Define CRF
        self.crf = CRF(tagset_size)

    def forward(self, tokens, attn_masks):
        encoded_layers = self.encoder(tokens, attention_mask=attn_masks)
        #print("hidden states shape: ",encoded_layers.last_hidden_state.size())
        hidden_states = encoded_layers.last_hidden_state
        #print("hidden states shape after meaning: ",hidden_states.size())
        lstm_out, final_memory_state = self.lstm1(hidden_states)
        #print("lstm1 out shape: ",lstm_out.size())
        #attn_out, attention_weights = self.self_attention(lstm_out)
        #print("attn out shape: ",attn_out.size())
        #lstm_out, _ = self.lstm2(lstm_out, final_memory_state)
        #print("lstm2 out shape: ",lstm_out.size())
        tag_space = self.hidden2tag(lstm_out)
        #print("h2t out: ",tag_space.size())
        tag_scores = self.crf.decode(tag_space)

        return torch.tensor(tag_scores), tag_space

def main():
    args = parse_args()

    # Define the mapping from index to tag
    idx2tag = {0: 'B', 1: 'I', 2: 'O', 3: 'E'}
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}

    # Detect device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = EDUPredictor(tagset_size=len(idx2tag.keys()), hidden_dim=args.hidden_dim, max_length = args.max_length).to(device)

    if args.prepare:
        # Use the function
        train_df = preprocess_RST_Discourse_dataset(os.path.join(args.rst_dir, "train"), tag2idx, args, model)
        # Save the dataframe to a csv file for further use
        train_df.to_csv(os.path.join(args.rst_dir, 'preprocessed_data_train.csv'), index=False)
        test_df = preprocess_RST_Discourse_dataset(os.path.join(args.rst_dir, "test"), tag2idx, args, model)
        # Save the dataframe to a csv file for further use
        test_df.to_csv(os.path.join(args.rst_dir, 'preprocessed_data_test.csv'), index=False)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.train:
         # Convert data to PyTorch tensors and move to the device
        train_data = pd.read_csv(os.path.join(args.rst_dir, 'preprocessed_data_train.csv'))
        
        train_data['Text'] = train_data['Text'].tolist()
        for i in range(len(train_data['Text'])):
            #print(train_data['Text'].iloc[i])
            train_data['Text'].iloc[i] =  np.array(ast.literal_eval(train_data['Text'].iloc[i]))
            train_data['Text'].iloc[i] = [int(item) for item in train_data['Text'].iloc[i]]
        train_inputs = torch.tensor(np.array(train_data['Text'].tolist()))
 
        attention_masks = train_data['Attention Mask' ].tolist()
        for i in range(len(train_data['Attention Mask'])):
            train_data['Attention Mask'].iloc[i] =  np.array(ast.literal_eval(train_data['Attention Mask'].iloc[i]))
            train_data['Attention Mask'].iloc[i] = [int(item) for item in train_data['Attention Mask'].iloc[i]]
        attention_masks = torch.tensor(np.array(train_data['Attention Mask' ].tolist()))
        attention_masks =  manual_batching(attention_masks, args.batch_size)
        #train_tuples = list(zip(train_inputs, attention_masks))
        #train_tuples = torch.stack((train_inputs, attention_masks), dim=0).to(device) #[2, 342, 18432]
        #train_tuples = train_inputs
        #torch.tensor(train_tuples, dtype=torch.long).to(device)
        train_labels = train_data['BIOE'].tolist()
        train_labels = [ast.literal_eval(label_list) for label_list in train_labels]
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

        # Create DataLoader for training data
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # Training loop
        for epoch in tqdm(range(args.epochs), desc='Epochs'):
            epoch_loss = 0.0
            epoch_f1 = [0.0] *4
            model.train()  # Set model to training mode

            # Create a tqdm progress bar for the inner loop (train_loader)
            train_loader_tqdm = tqdm(enumerate(zip(train_loader, attention_masks)), total=len(train_loader), desc='Batches')
            for step, ((inputs, labels), attention_mask) in train_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                optimizer.zero_grad()  # Zero the gradients

                # Forward propagation
                tag_scores, emissions = model(inputs, attention_mask)

                scores = compute_f1_score_for_labels(tag_scores.detach().cpu().numpy().flatten(), labels.detach().cpu().numpy().flatten(), labels= [int(key) for key in idx2tag.keys()])
                # Compute the loss
                loss = -model.crf(emissions, labels)

                # Backward propagation
                loss.backward()

                # Gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimization step
                optimizer.step()
                #print(scores)
                epoch_loss += loss.item()
                for i in range (len(epoch_f1)):
                    epoch_f1[i] += scores[i]['F1 Score']
                # Update the tqdm progress bar with the current loss value
                running_f1 = [epoch_f1/(step+1) for item in epoch_f1]
                train_loader_tqdm.set_postfix({f"f1 scores for tag B: {running_f1[0]:.4f}, tag I: {running_f1[1]:.4f}, tag O: {running_f1[2]:.4f}, tag E: {running_f1[3]:.4f} and Loss": epoch_loss / (step + 1)})

            # Update the outer tqdm progress bar with the current epoch loss value
            tqdm.write(f'Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / len(train_loader):.4f}, F1: {[epoch_f1/len(train_loader) for item in epoch_f1]:.4f}')

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
            test_tag_scores = model(test_inputs).detach().cpu().numpy().flatten()
            test_pred_tags = test_tag_scores.detach().cpu().numpy().flatten()
            test_tags = test_labels.detach().cpu().numpy().flatten()
            #test_pred = model.crf.decode(test_tag_scores)
            #scores = compute_f1_score_for_labels(test_tag_scores.detach().cpu().numpy().flatten(), test_labels.detach().cpu().numpy().flatten(), labels= idx2tag.keys())
            # Flatten both labels and predictions
            #test_tags = [idx2tag[i] for row in test_labels for i in row]
            #test_pred_tags = [idx2tag[i] for row in test_pred.detach().cpu().numpy().flatten() for i in row]

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
