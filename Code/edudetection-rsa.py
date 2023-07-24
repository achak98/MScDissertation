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
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2ForMaskedLM
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore') 

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

def compute_f1_score_for_labels(y_true, y_pred, labels):
    # y_true: Ground truth labels (list of lists)
    # y_pred: Predicted labels (list of lists)
    # labels: List of unique labels in your dataset
    # Flatten the ground truth and predicted labels
    #y_true_flat = [label for sublist in y_true for label in sublist]
    #y_pred_flat = [label for sublist in y_pred for label in sublist]

    # Compute precision, recall, and F1 score for each label
    precision, recall, f1_score_for_label, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    overall_f1 = (f1_score_for_label[0]*support[0] + f1_score_for_label[1]*support[1])/(support[0]+support[1]) #+ f1_score_for_label[2]*support[2] + f1_score_for_label[3]*support[3])/(support[0]+support[1]+support[2]+support[3])
    correct = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)
    total = len(y_true)
    accuracy = correct / total * 100.0
    # Create a dictionary to store the results for each label
    label_scores = {}
    for i, label in enumerate(labels):
        label_scores[label] = {
            'Precision': precision[i],
            'Recall': recall[i],
            'F1 Score': f1_score_for_label[i]
        }

    return label_scores, accuracy, overall_f1
    
def parse_args():
    parser = argparse.ArgumentParser('EDU segmentation toolkit 1.0')
    parser.add_argument('--prepare', action='store_true',
                        help='preprocess the RST-DT data and create the vocabulary')
    parser.add_argument('--train', action='store_true', help='train the segmentation model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--segment', action='store_true', help='segment new files or input text')
    parser.add_argument('--get_embeddings_anyway', action='store_true', help='train the segmentation model')
    parser.add_argument('--regex_pattern', default= r'\b\w+\b|[.,:\n&!]')
    parser.add_argument('--max_length', type=int, default= 2048)
    parser.add_argument('--learning_rate', type=float,
                                default=3e-4, help='learning rate')
    parser.add_argument('--device', type=int,
                                default=0, help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                                default=1e-3, help='weight decay')
    parser.add_argument('--dropout', type=float,
                                default=0.2, help='weight decay')
    parser.add_argument('--window_size', type=int,
                                default=5, help='die ya whoreses')
    parser.add_argument('--batch_size', type=int,
                                default=1, help='batch size')
    parser.add_argument('--epochs', type=int,
                                default=10, help='train epochs')
    parser.add_argument('--seed', type=int,
                                default=42, help='the random seed')
    parser.add_argument('--tagset_size', type=int,
                                default=2, help='number of tags in the tagset')
    parser.add_argument('--hidden_dim', type=int,
                                default=768, help='hidden size')
    parser.add_argument('--max_grad_norm', type=float,
                                default=3.0, help='gradient norm')
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

def find_sequence_spans(sents, edus, model, args):
    sequence_spans = []
    idx_edu = 0

    for idx_sents, sent in enumerate(sents):
        tokenised_sent = model.tokeniser(sent)["input_ids"]
        loop = True
        i = 0
        start_index = None
        while(loop):
            if(i>=len(tokenised_sent) or idx_edu >= len(edus)):
                loop = False
                break
            edu = edus[idx_edu]
            tokenised_edu = model.tokeniser(edu)["input_ids"]
            tokenised_edu = tokenised_edu[1:-1]
            target_length = len(tokenised_edu)
            if target_length == 0 :
                idx_edu+=1
            else:
                if tokenised_edu[0] == tokenised_sent[i]:
                    broken = False
                    start_index = i
                    idx_edu+=1
                    potential_end = i + target_length -1
                    if potential_end > len(tokenised_sent)-1 :
                        """print(f"sent: {sent} edu: {edu} \n \
                              tokenised sent: {tokenised_sent} \n \
                              tokenised edu: {tokenised_edu} \n \
                              tonkenised_edu_corres_word: {tokenised_edu[len(tokenised_sent)-2 - i]}\n \
                              tonkenised_edu_corres_word in if: {tokenised_edu[len(tokenised_sent)-2 - i]}\n \
                              tokenised_sent_last_word in if: {tokenised_edu[len(tokenised_sent)-2 - i]}\n \
                              tokenised_sent_last_word: {tokenised_sent[-2]}")"""
                        if tokenised_edu[len(tokenised_sent)-2 - i] == tokenised_sent[-2]:
                            potential_end = len(tokenised_sent)-2 - i
                            broken = True
                        else:
                            break
                    if tokenised_sent[potential_end] == tokenised_edu[-1] or broken:
                        #print("end matched")
                        end_index = potential_end
                        i = end_index
                        sequence_spans.append((start_index, end_index, idx_sents))
                start_index = None
                i+=1

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
            sents = text.split('\n')
            sents = [sent.replace("'", " ' ").replace("(", " ( ").replace(")", " )' ").replace("\"", " \" ").replace("-", " - ").replace(",", " , ").replace(".", " . ") for sent in sents]
            edus = [edu.replace("'", " ' ").replace("\"", " \" ").replace("(", " ( ").replace(")", " )' ").replace("-", " - ").replace(",", " , ").replace(".", " . ") for edu in edus]
            edus = [seq.strip() for seq in edus]
            
            sequence_spans = find_sequence_spans(sents, edus, model, args)
            idx_seq_spans = 0
            avg_len = 0
            cnt = 0
            for idx_sents, sent in enumerate(sents):
                if(len(sent)==0):
                  continue
                avg_len+= len(sent)
                cnt+=1
                tokenised_sent = model.tokeniser(sent, padding="max_length", return_attention_mask=True, max_length = 2048)
                input_ids = tokenised_sent["input_ids"]
                attn_mask = tokenised_sent["attention_mask"]
                
                loop = True
                IO_tags = [tag2idx["O"]] * len(input_ids)
                while (loop):
                    if(idx_seq_spans >= len(sequence_spans) or sequence_spans[idx_seq_spans][2] != idx_sents):
                        loop = False
                        break
                    span = sequence_spans[idx_seq_spans]
                    if(span[2] == idx_sents):
                        for i in range(span[0],span[1]+1):
                            IO_tags[i] = tag2idx['I']
                    idx_seq_spans+=1
                data.append((input_ids[:args.max_length], attn_mask[:args.max_length], IO_tags[:args.max_length]))
        if(len(edus)-1 != len(sequence_spans)):
            print(f"messed up file: {txt_file}, detected: {len(sequence_spans)}, total: {len(edus)-1}")
            messed_up_ones.append(txt_file)
    df = pd.DataFrame(data, columns=['Sentence', 'Attention Mask', 'IO'])
    print(messed_up_ones)
    return df

class EDUPredictor(nn.Module):
    def __init__(self, args):
        super(EDUPredictor, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.tagset_size = args.tagset_size
        self.max_length = args.max_length
        self.window_size = args.window_size
        self.transformer_architecture = 'microsoft/deberta-v3-base' #'microsoft/deberta-v3-small' mlcorelib/debertav2-base-uncased microsoft/deberta-v2-xlarge
        self.config = AutoConfig.from_pretrained(self.transformer_architecture, output_hidden_states=True)
        self.config.max_position_embeddings = self.max_length
        self.tokeniser = AutoTokenizer.from_pretrained(self.transformer_architecture, max_length=self.config.max_position_embeddings, padding="max_length", return_attention_mask=True)

        # Define BiLSTM 1
        self.lstm1 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(args.dropout) 
        # Attention weight computation layer
        self.attention_weights = nn.Linear(args.hidden_dim * 3, 1)
        # Define BiLSTM 2
        self.lstm2 = nn.LSTM(self.hidden_dim*2, self.tagset_size, num_layers=1, bidirectional=True)
        self.dropout2 = nn.Dropout(args.dropout)  
        """self.regressor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear( hidden_dim//2,  hidden_dim),
            nn.GELU()
        )"""
        # Define MLP
        """self.hidden2tag = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//16),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim//16, self.tagset_size),
            nn.GELU(),
            nn.Dropout(0.3)
        )"""
        #print("tagset_size: ",tagset_size)
        # Define CRF
        self.crf = CRF(self.tagset_size)
    
    def similarity(self, hi, hj):
        # Concatenate the hidden representations
        h_concat = torch.cat([hi, hj, hi * hj], dim=-1)
        return self.attention_weights(h_concat)
    def forward(self, embeddings):
 
        lstm_out, _ = self.lstm1(embeddings)
        lstm_out = self.dropout1(lstm_out)
        hidden_dim_size = lstm_out.size(-1)
        batch_size, seq_length, hidden_dim = lstm_out.size()
        first_half = lstm_out[:, :, : hidden_dim_size// 2]
        second_half = lstm_out[:, :, hidden_dim_size // 2:]
        output_sum = first_half + second_half
        # Initialize attention vector tensor
        attention_vectors = torch.zeros_like(output_sum)
        for i in range(seq_length):
            # Define the start and end positions of the window
            start_pos = max(0, i - self.window_size)
            end_pos = min(seq_length, i + self.window_size + 1)
            
            # Compute similarity between the current word and nearby words
            similarity_scores = torch.cat([self.similarity(output_sum[:, i], output_sum[:, j]) for j in range(start_pos, end_pos)], dim=1)
            #print("similarity_scores: ",similarity_scores.size())
            #similarity scores has all s(i,j)s from start pos to end pos of window
            # Apply softmax to obtain attention weights
            attention_weights = torch.nn.functional.softmax(similarity_scores, dim=-1) #this has all alpha(i,j)s
            #print("attention_weights: ",attention_weights.size())
            # Compute the attention vector as a weighted sum of nearby words
            #print("_sum : ",_sum.size())
            #print("output_sum[:, start_pos:end_pos, :].permute(0,2,1: ", output_sum[:, start_pos:end_pos, :].permute(0,2,1).size())
            #print("output_sum[:, start_pos:end_pos, :]: ", output_sum[:, start_pos:end_pos, :].size())
            #print("attention_weights: ", attention_weights.size())
            attention_vector = torch.sum((output_sum[:, start_pos:end_pos, :].permute(2,0,1) * attention_weights).permute(1,2,0), dim=1)
            #print("attention_vector: ",attention_vector.size())
            # Store the attention vector for the current word
            attention_vectors[:,i] = attention_vector.squeeze(1) #(seqlen,hiddim)

        #print("attention_vectors: ",attention_vectors.size())
        #print("output_sum: ",output_sum.size())
        # Concatenate the original LSTM output and the attention vectors
        lstm_output_with_attention = torch.cat([output_sum, attention_vectors], dim=-1)
        #print("lstm_output_with_attention: ", lstm_output_with_attention.size())
        #print(output_sum.size())
        #print(output_sum.size())
        lstm_out, _ = self.lstm2(lstm_output_with_attention)

        #lstm_out, _ = self.lstm2(output_sum)
        lstm_out = self.dropout2(lstm_out)
        #print(lstm_out.size())
        #tag_space = self.hidden2tag(lstm_out)
        #print("size of tag_space: ", tag_space.size())
        hidden_dim_size = lstm_out.size(-1)
        first_half = lstm_out[:, :, : hidden_dim_size// 2]
        second_half = lstm_out[:, :, hidden_dim_size // 2:]
        
        #print("first_half: ", first_half.size())
        # Sum the two halves together along the last dimension
        output_sum = first_half + second_half
        #print(output_sum.size())
        tag_scores = self.crf.decode(output_sum)

        return torch.tensor(tag_scores), output_sum

def validation(args,idx2tag,model, val_embeddings, val_labels):
    # Detect device (CPU or CUDA)
    torch.cuda.empty_cache()
    device = f"cuda:{args.device}"
    outputs = torch.empty((len(val_labels),args.max_length), dtype=torch.float).to(device)
    val_embeddings = torch.tensor(val_embeddings).to(device)
    val_labels = val_labels.to(device)
    #print("embeddings in val: ",val_embeddings)

    # Evaluation
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Predict output for test set
        val_embeddings = val_embeddings.to(torch.float)
        for i, (embedding, test_label) in enumerate(zip(val_embeddings,val_labels)):
            #print("embedding in val: ",embedding.size())
            #print("unsq embedding in val: ",embedding.unsqueeze(0).size())
            output, _ = model(embedding.unsqueeze(0))
            outputs[i] = output.squeeze()
        val_labels = val_labels.to(device)
        #outputs, emissions = model(val_embeddings)
        test_pred_tags = outputs.detach().to(torch.long).cpu().numpy().flatten()
        test_tags = val_labels.detach().cpu().numpy().flatten()
        #print("idx2tag.keys(): ",idx2tag.keys())
        #print("test_pred_tags: ",test_pred_tags)
        #print("test_tags: ",test_tags)
        scores, accuracy_score, overall_f1 = compute_f1_score_for_labels(test_pred_tags, test_tags, labels= [int(key) for key in idx2tag.keys()])
        epoch_f1 = [0.0]*2
        epoch_pre = [0.0]*2
        epoch_re = [0.0]*2
        for i in range (len(epoch_f1)):
                epoch_f1[i] += scores[i]['F1 Score']
                epoch_pre[i] += scores[i]['Precision']
                epoch_re[i] += scores[i]['Recall']
        return accuracy_score, epoch_pre, epoch_f1, epoch_re, overall_f1

def getValData(args, model):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    val_data = pd.read_csv(os.path.join(args.rst_dir, 'preprocessed_data_test.csv'))
        
    val_data['Sentence'] = val_data['Sentence'].tolist()
    for i in range(len(val_data['Sentence'])):
        #print(test_data['Sentence'].iloc[i])
        val_data['Sentence'].iloc[i] =  np.array(ast.literal_eval(val_data['Sentence'].iloc[i]))
        val_data['Sentence'].iloc[i] = [int(item) for item in val_data['Sentence'].iloc[i]]
    test_inputs = torch.cat((torch.tensor(np.array(val_data['Sentence'].tolist()))[:4], torch.tensor(np.array(val_data['Sentence'].tolist()))[-4:]), dim=0)
    attention_masks = val_data['Attention Mask' ].tolist()
    for i in range(len(val_data['Attention Mask'])):
        val_data['Attention Mask'].iloc[i] =  np.array(ast.literal_eval(val_data['Attention Mask'].iloc[i]))
        val_data['Attention Mask'].iloc[i] = [int(item) for item in val_data['Attention Mask'].iloc[i]]
    attention_masks = torch.cat((torch.tensor(np.array(val_data['Attention Mask' ].tolist()))[:4], torch.tensor(np.array(val_data['Attention Mask' ].tolist()))[-4:]), dim=0)
    
    val_labels = val_data['IO'].tolist()
    val_labels = [ast.literal_eval(label_list) for label_list in val_labels]
    val_labels = torch.cat(((torch.tensor(val_labels, dtype=torch.long).to(device))[:4], torch.tensor(val_labels, dtype=torch.long).to(device))[-4:], dim=0)
    #print("val_labels: ",val_labels)
    print("getting empty embeddings tensor")
    #print("args.get_embeddings_anyway in val: ", args.get_embeddings_anyway)
    if (not args.get_embeddings_anyway) and os.path.exists(os.path.join(args.rst_dir,'embeddings_val.pt')):
        val_embeddings = torch.load(os.path.join(args.rst_dir,'embeddings_val.pt'))
        print(f"val embeddings loaded from {os.path.join(args.rst_dir,'embeddings_val.pt')}")
    else:
        print(f"getting val embeddings...")
        val_embeddings = torch.empty((len(test_inputs),args.max_length,args.hidden_dim), dtype=torch.float64).to(device)
        print("init model")
        with torch.no_grad():
            input_ids = test_inputs.to(device)
            #print("input_ids in val: ",input_ids)
            #print("input_ids shape: ",input_ids.size())
            attention_masks = attention_masks.to(device) 
            encoder = AutoModel.from_pretrained(model.transformer_architecture, config=model.config)
            encoder = encoder.to(device)
            print("starting tqdm")
            for i in tqdm(range(len(input_ids))):
                input_id = input_ids[i].unsqueeze(0)
                attention_mask = attention_masks[i].unsqueeze(0)
                # Obtain deberta embeddings for the current item
                outputs = encoder(input_id, attention_mask)
                val_embeddings[i] = torch.tensor(outputs.last_hidden_state).squeeze()
            #print("embeddings.size(): ",val_embeddings.size())
        torch.save(val_embeddings, os.path.join(args.rst_dir,'embeddings_val.pt'))
    torch.cuda.empty_cache()
    return val_embeddings,val_labels

def main():
    args = parse_args()

    # Define the mapping from index to tag
    idx2tag = {0: 'I', 1: 'O'} #    , 2: 'O', 3: 'E'}
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}

    # Detect device (CPU or CUDA)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = EDUPredictor(args).to(device)

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
        
        train_data['Sentence'] = train_data['Sentence'].tolist()
        for i in range(len(train_data['Sentence'])):
            #print(train_data['Sentence'].iloc[i])
            train_data['Sentence'].iloc[i] =  np.array(ast.literal_eval(train_data['Sentence'].iloc[i]))
            train_data['Sentence'].iloc[i] = [int(item) for item in train_data['Sentence'].iloc[i]]
        train_inputs = torch.tensor(np.array(train_data['Sentence'].tolist()))
 
        attention_masks = train_data['Attention Mask' ].tolist()
        for i in range(len(train_data['Attention Mask'])):
            train_data['Attention Mask'].iloc[i] =  np.array(ast.literal_eval(train_data['Attention Mask'].iloc[i]))
            train_data['Attention Mask'].iloc[i] = [int(item) for item in train_data['Attention Mask'].iloc[i]]
        attention_masks = torch.tensor(np.array(train_data['Attention Mask' ].tolist()))
 
        train_labels = train_data['IO'].tolist()
        train_labels = [ast.literal_eval(label_list) for label_list in train_labels]
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

        """if (not args.get_embeddings_anyway) and os.path.exists(os.path.join(args.rst_dir,'embeddings_train.pt')):
            embeddings = torch.load(os.path.join(args.rst_dir,'embeddings_train.pt'))
            print(f"train embeddings loaded from {os.path.join(args.rst_dir,'embeddings_train.pt')}")
        else:
            print(f"getting train embeddings...")
            embeddings = torch.empty((len(train_inputs),args.max_length,args.hidden_dim), dtype=torch.float64).to(device)
            print("init model")
            with torch.no_grad():
                input_ids = train_inputs.to(device)
                print("input_ids shape: ",input_ids.size())
                attention_masks = attention_masks.to(device) 
                encoder = AutoModel.from_pretrained(model.transformer_architecture, config=model.config)
                encoder = encoder.to(device)
                print("starting tqdm")
                for i in tqdm(range(len(input_ids))):
                    input_id = input_ids[i].unsqueeze(0)
                    attention_mask = attention_masks[i].unsqueeze(0)
                    # Obtain deberta embeddings for the current item
                    outputs = encoder(input_id, attention_mask)
                    embeddings[i] = torch.tensor(outputs.last_hidden_state).squeeze()
                #print("embeddings.size(): ",embeddings.size())
            torch.save(embeddings, os.path.join(args.rst_dir,'embeddings_train.pt'))"""
        torch.cuda.empty_cache()
        val_embeddings, val_labels = getValData(args, model)

        #embeddings = torch.tensor(embeddings).to(device)
        # Create DataLoader for training data
        #print(train_labels.size())
        #print(embeddings.size())
        
        train_inputs = manual_batching(train_inputs, args.batch_size)
        attention_masks = manual_batching(attention_masks, args.batch_size)
        train_labels = manual_batching(train_labels, args.batch_size)

        print("starting training")
        # Training loop
        for epoch in tqdm(range(args.epochs), desc='Epochs'):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_f1 = [0.0] * 2
            epoch_pre = [0.0] * 2
            epoch_re = [0.0] * 2
            epoch_overall_f1 = 0.0
            model = model.to(device)
            encoder = AutoModel.from_pretrained(model.transformer_architecture, config=model.config)
            encoder = encoder.to(device)
            model.train()  # Set model to training mode

            # Create a tqdm progress bar for the inner loop (train_loader)
            train_loader_tqdm = tqdm(train_inputs, total=len(train_inputs), desc='Batches')
            for step, (input_id,attention_mask,labels) in enumerate(zip(train_loader_tqdm, attention_masks, train_labels)):
                input_id = input_id.to(device)
                #print("input_id: ",input_id.size())
                attention_mask = attention_mask.to(device) 
                embeddings = encoder(input_id, attention_mask).last_hidden_state
                
                inputs = embeddings.to(torch.float) #.to(device)
                labels = labels.to(device)
                #print("inputs: ",inputs.size())
                #print("labels: ",labels.size())
                #print(f"type of inputs tensor: {inputs.dtype}, and type of labels tensor: {labels.dtype}") 

                optimizer.zero_grad()  # Zero the gradients
                #print("inputs in train: ",inputs.size())
                # Forward propagation
                tag_scores, emissions = model(inputs)

                scores, accuracy_score, overall_f1 = compute_f1_score_for_labels(tag_scores.detach().to(torch.long).cpu().numpy().flatten(), labels.detach().cpu().numpy().flatten(), labels= [int(key) for key in idx2tag.keys()])
                # Compute the loss
                loss = -model.crf(emissions, labels, reduction = 'token_mean')

                # Backward propagation
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimization step
                optimizer.step()
                #print(scores)
                epoch_loss += loss.item()
                epoch_acc += accuracy_score
                epoch_overall_f1 += overall_f1
                for i in range (len(epoch_f1)):
                    epoch_f1[i] += scores[i]['F1 Score']
                    epoch_pre[i] += scores[i]['Precision']
                    epoch_re[i] += scores[i]['Recall']
                # Update the tqdm progress bar with the current loss value
                running_f1 = [item/(step+1) for item in epoch_f1]
                train_loader_tqdm.set_postfix ({f"f1 scores for tag I: {running_f1[0]:.3f}, tag O: {running_f1[1]:.3f}, Acc: {(epoch_acc/(step+1)):.3f} and Loss": epoch_loss / (step + 1)})#({f"f1 scores for tag B: {running_f1[0]:.3f}, tag I: {running_f1[1]:.3f}, tag O: {running_f1[2]:.3f}, tag E: {running_f1[3]:.3f}, Acc: {(epoch_acc/(step+1)):.3f} and Loss": epoch_loss / (step + 1)})
            epoch_f1 = [item/len(train_inputs) for item in epoch_f1]
            epoch_pre = [item/len(train_inputs) for item in epoch_pre]
            epoch_re = [item/len(train_inputs) for item in epoch_re]
            epoch_acc = epoch_acc/len(train_inputs)
            epoch_overall_f1 = epoch_overall_f1/len(train_inputs)
            # Update the outer tqdm progress bar with the current epoch loss value
            val_accuracy_score, val_epoch_pre, val_epoch_f1, val_epoch_re, val_overall_f1 = validation(args,idx2tag,model, val_embeddings, val_labels)
            
            #print(f'F1 scores for tag B: {epoch_f1[0]:.3f}, tag I: {epoch_f1[1]:.3f}, tag O: {epoch_f1[2]:.3f}, tag E: {epoch_f1[3]:.3f}')
            tqdm.write(f'-------------------------------------------------------------------------------------------------------------------------------------\n \
                        Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / len(train_inputs):.3f}\n \
                        Acc: {epoch_acc:.3f} and Test Acc: {val_accuracy_score:.3f}\n \
                        Overall F1: {overall_f1:.3f} Val Overall F1: {val_overall_f1:.3f} \n\
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag I: \n\
                        Train F1: {epoch_f1[0]:.3f}, Val F1: {val_epoch_f1[0]:.3f} \n \
                        Train Precision: {epoch_pre[0]:.3f}, Val Precision: {val_epoch_pre[0]:.3f} \n \
                        Train Recall: {epoch_re[0]:.3f}, Val Recall: {val_epoch_re[0]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag O: \n\
                        Train F1: {epoch_f1[1]:.3f}, Val F1: {val_epoch_f1[1]:.3f} \n \
                        Train Precision: {epoch_pre[1]:.3f}, Val Precision: {val_epoch_pre[1]:.3f} \n \
                        Train Recall: {epoch_re[1]:.3f}, Val Recall: {val_epoch_re[1]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------')
            """tqdm.write(f'-------------------------------------------------------------------------------------------------------------------------------------\n \
                        Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / len(train_loader):.3f}\n \
                        Acc: {epoch_acc:.3f} and Test Acc: {val_accuracy_score:.3f}\n \
                        Overall F1: {overall_f1:.3f} Val Overall F1: {val_overall_f1:.3f} \n\
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag B: \n\
                        Train F1: {epoch_f1[0]:.3f}, Val F1: {val_epoch_f1[0]:.3f} \n \
                        Train Precision: {epoch_pre[0]:.3f}, Val Precision: {val_epoch_pre[0]:.3f} \n \
                        Train Recall: {epoch_re[0]:.3f}, Val Recall: {val_epoch_re[0]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag I: \n\
                        Train F1: {epoch_f1[1]:.3f}, Val F1: {val_epoch_f1[1]:.3f} \n \
                        Train Precision: {epoch_pre[1]:.3f}, Val Precision: {val_epoch_pre[1]:.3f} \n \
                        Train Recall: {epoch_re[1]:.3f}, Val Recall: {val_epoch_re[1]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag O: \n\
                        Train F1: {epoch_f1[2]:.3f}, Val F1: {val_epoch_f1[2]:.3f} \n \
                        Train Precision: {epoch_pre[2]:.3f}, Val Precision: {val_epoch_pre[2]:.3f} \n \
                        Train Recall: {epoch_re[2]:.3f}, Val Recall: {val_epoch_re[2]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag E: \n\
                        Train F1: {epoch_f1[3]:.3f}, Val F1: {val_epoch_f1[3]:.3f} \n \
                        Train Precision: {epoch_pre[3]:.3f}, Val Precision: {val_epoch_pre[3]:.3f} \n \
                        Train Recall: {epoch_re[3]:.3f}, Val Recall: {val_epoch_re[3]:.3f} \n \
                        -------------------------------------------------------------------------------------------------------------------------------------')"""

        # Save the trained model
        model_path = os.path.join(args.model_dir, 'edu_segmentation_model_rsa.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to: {model_path}")
    torch.cuda.empty_cache()
    if args.evaluate:
        test_data = pd.read_csv(os.path.join(args.rst_dir, 'preprocessed_data_test.csv'))
        
        test_data['Sentence'] = test_data['Sentence'].tolist()
        for i in range(len(test_data['Sentence'])):
            #print(test_data['Sentence'].iloc[i])
            test_data['Sentence'].iloc[i] =  np.array(ast.literal_eval(test_data['Sentence'].iloc[i]))
            test_data['Sentence'].iloc[i] = [int(item) for item in test_data['Sentence'].iloc[i]]
        test_inputs = torch.tensor(np.array(test_data['Sentence'].tolist()))
 
        attention_masks = test_data['Attention Mask' ].tolist()
        for i in range(len(test_data['Attention Mask'])):
            test_data['Attention Mask'].iloc[i] =  np.array(ast.literal_eval(test_data['Attention Mask'].iloc[i]))
            test_data['Attention Mask'].iloc[i] = [int(item) for item in test_data['Attention Mask'].iloc[i]]
        attention_masks = torch.tensor(np.array(test_data['Attention Mask' ].tolist()))
 
        test_labels = test_data['IO'].tolist()
        test_labels = [ast.literal_eval(label_list) for label_list in test_labels]
        test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
        print("getting empty embeddings tensor")
        if (not args.get_embeddings_anyway) and os.path.exists(os.path.join(args.rst_dir,'embeddings_test.pt')):
            embeddings = torch.load(os.path.join(args.rst_dir,'embeddings_test.pt'))
            print(f"test embeddings loaded from {os.path.join(args.rst_dir,'embeddings_test.pt')}")
        else:
            print(f"getting test embeddings...")
            embeddings = torch.empty((len(test_inputs),args.max_length,args.hidden_dim), dtype=torch.float64).to(device)
            print("init model")
            with torch.no_grad():
                input_ids = test_inputs.to(device)
                print("input_ids shape: ",input_ids.size())
                attention_masks = attention_masks.to(device) 
                encoder = AutoModel.from_pretrained(model.transformer_architecture, config=model.config)
                encoder = encoder.to(device)
                print("starting tqdm")
                for i in tqdm(range(len(input_ids))):
                    input_id = input_ids[i].unsqueeze(0)
                    attention_mask = attention_masks[i].unsqueeze(0)
                    # Obtain deberta embeddings for the current item
                    outputs = encoder(input_id, attention_mask)
                    embeddings[i] = torch.tensor(outputs.last_hidden_state).squeeze()
                print("embeddings.size(): ",embeddings.size())
            torch.save(embeddings, os.path.join(args.rst_dir,'embeddings_test.pt'))
        torch.cuda.empty_cache()

        device = torch.device(f"cuda:{args.device}")
        embeddings = torch.tensor(embeddings).to(device)
    
        # Load the trained model
        model_path = os.path.join(args.model_dir, 'edu_segmentation_model_rsa.pt')
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        print(f"Loaded trained model from: {model_path}")

        # Evaluation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Predict output for test set
            embeddings = embeddings.to(torch.float)
            test_labels = test_labels.to(device)
            outputs, emissions = model(embeddings)
            loss = -model.crf(emissions, test_labels)
            test_pred_tags = outputs.detach().cpu().numpy().flatten()
            test_tags = test_labels.detach().cpu().numpy().flatten()
            scores, accuracy_score, test_overall_f1 = compute_f1_score_for_labels(test_pred_tags, test_tags, labels= [int(key) for key in idx2tag.keys()])
            #test_pred = model.crf.decode(test_tag_scores)
            #scores = compute_f1_score_for_labels(test_tag_scores.detach().cpu().numpy().flatten(), test_labels.detach().cpu().numpy().flatten(), labels= idx2tag.keys())
            # Flatten both labels and predictions
            #test_tags = [idx2tag[i] for row in test_labels for i in row]
            #test_pred_tags = [idx2tag[i] for row in test_pred.detach().cpu().numpy().flatten() for i in row]
            print("\n \n test_pred_tags: ",test_pred_tags)
            print("\n \n test_pred_tags: ",type(test_pred_tags))
            print("\n \n test_tags: ", test_tags)
            print("\n \n test_tags: ", type(test_tags))
            # Compute evaluation metrics
            #accuracy = accuracy_score(test_tags, test_pred_tags)
            #precision = precision_score(test_tags, test_pred_tags)
            #recall = recall_score(test_tags, test_pred_tags)
            #f1 = f1_score(test_tags, test_pred_tags)
            epoch_f1 = [0.0]*2
            epoch_pre = [0.0]*2
            epoch_re = [0.0]*2
            for i in range (len(epoch_f1)):
                    epoch_f1[i] += scores[i]['F1 Score']
                    epoch_pre[i] += scores[i]['Precision']
                    epoch_re[i] += scores[i]['Recall']
            print(f'-------------------------------------------------------------------------------------------------------------------------------------\n \
                Acc: {accuracy_score:.3f} Overall F1: {test_overall_f1:.3f} \n\
                -------------------------------------------------------------------------------------------------------------------------------------\n \
                for tag I: \n\
                Test F1: {epoch_f1[0]:.3f}\n \
                Test Precision: {epoch_pre[0]:.3f}\n \
                Test Recall: {epoch_re[0]:.3f}\n \
                -------------------------------------------------------------------------------------------------------------------------------------\n \
                for tag O: \n\
                Test F1: {epoch_f1[1]:.3f}\n \
                Test Precision: {epoch_pre[1]:.3f}\n \
                Test Recall: {epoch_re[1]:.3f}\n \
                -------------------------------------------------------------------------------------------------------------------------------------')
            """print(f'-------------------------------------------------------------------------------------------------------------------------------------\n \
                        Acc: {accuracy_score:.3f} Overall F1: {test_overall_f1:.3f} \n\
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag B: \n\
                        Test F1: {epoch_f1[0]:.3f}\n \
                        Test Precision: {epoch_pre[0]:.3f}\n \
                        Test Recall: {epoch_re[0]:.3f}\n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag I: \n\
                        Test F1: {epoch_f1[1]:.3f}\n \
                        Test Precision: {epoch_pre[1]:.3f}\n \
                        Test Recall: {epoch_re[1]:.3f}\n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag O: \n\
                        Test F1: {epoch_f1[2]:.3f}\n \
                        Test Precision: {epoch_pre[2]:.3f}\n \
                        Test Recall: {epoch_re[2]:.3f}\n \
                        -------------------------------------------------------------------------------------------------------------------------------------\n \
                        for tag E: \n\
                        Test F1: {epoch_f1[3]:.3f}\n \
                        Test Precision: {epoch_pre[3]:.3f}\n \
                        Test Recall: {epoch_re[3]:.3f}\n \
                        -------------------------------------------------------------------------------------------------------------------------------------')"""

            with open(os.path.join(args.result_dir, "edu_results.txt"), 'w') as file:
                file.write(f'Loss: {loss.item():.3f}, f1 scores for tag I: {epoch_f1[0]:.3f}, tag O: {epoch_f1[1]:.3f}, and Acc: {accuracy_score}:.3f')
                #file.write(f'Loss: {loss.item():.3f}, f1 scores for tag B: {epoch_f1[0]:.3f}, tag I: {epoch_f1[1]:.3f}, tag O: {epoch_f1[2]:.3f}, tag E: {epoch_f1[3]:.3f}, and Acc: {accuracy_score}:.3f')

            # Generate confusion matrix
            confusion_matrix = multilabel_confusion_matrix(test_tags, test_pred_tags)
            for i, matrix in enumerate(confusion_matrix):
                plt.figure()
                sns.heatmap(matrix, annot=True, fmt='d')
                plt.title(f'Confusion Matrix for tag {i}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(args.result_dir, 'confusion_matrix_edu.pdf'))

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
        seg_df = pd.DataFrame(data, columns=['Sentence', 'IO'])
        seg_df.to_csv('preprocessed_data_test.csv', index=False)
if __name__ == '__main__':
    main()
