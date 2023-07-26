from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor
from argminer.evaluation import inference
from argminer.config import LABELS_MAP_DICT
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
from torchcrf import CRF 
from tqdm import tqdm
# set path to data source
path = '/Users/ac/Desktop/unicode/Dissertation/Data/ArgumentAnnotatedEssays-2.0'

import os

def trim_lines_in_files(directory_path):
    # List all files in the directory
    file_list = os.listdir(directory_path)
    #print("!!!!!!!!!!!!!!!!!file_list: ",file_list)
    for file_name in file_list:
        if(file_name.split(".")[-1]=="conf"):
            continue
        file_path = os.path.join(directory_path, file_name)
        
        # Skip directories (if any) and only process regular files
        if not os.path.isfile(file_path):
            continue

        # Read the content of the file with 'latin-1' encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()

        # Trim whitespace from each line
        trimmed_lines = [line.strip() for line in lines]

        # Save the trimmed content back to the file
        with open(file_path, 'w', encoding='latin-1') as file:
            file.write('\n'.join(trimmed_lines))


#trim_lines_in_files(os.path.join(path,"brat-project-final"))


processor = TUDarmstadtProcessor(path)
processor = processor.preprocess()
class EDUPredictor(nn.Module):
    def __init__(self):
        super(EDUPredictor, self).__init__()

        self.hidden_dim = 200
        self.tagset_size = 11
        self.max_length = 1024
        self.window_size = 5
        self.transformer_architecture = 'microsoft/deberta-v3-base' #'microsoft/deberta-v3-small' mlcorelib/debertav2-base-uncased microsoft/deberta-v2-xlarge
        self.config = AutoConfig.from_pretrained(self.transformer_architecture, output_hidden_states=True)
        self.config.max_position_embeddings = self.max_length
        self.tokeniser = AutoTokenizer.from_pretrained(self.transformer_architecture, max_length=self.config.max_position_embeddings, padding="max_length", return_attention_mask=True)

        # Define BiLSTM 1
        self.lstm1 = nn.LSTM(768, self.hidden_dim, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(0.1) 
        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim*2, self.hidden_dim//8),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_dim//8, self.tagset_size),
        nn.GELU(),
        nn.Dropout(0.1)
        )

        # Attention weight computation layer
        #self.attention_weights = nn.Linear(args.hidden_dim * 3, 1)
        #self.dropout2 = nn.Dropout(args.dropout) 
        # Define BiLSTM 2
        #self.lstm2 = nn.LSTM(self.hidden_dim, self.tagset_size, num_layers=1, bidirectional=True)
       # self.dropout3 = nn.Dropout(args.dropout)  

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
        output_sum = self.fc1(lstm_out)
        #print("output_sum.size(): ",output_sum.size())
        tag_scores = self.crf.decode(output_sum)
        for i in range(len(tag_scores[0])):
            if tag_scores[0][i] == 10:
                tag_scores[0][i] = -100
        return torch.tensor(tag_scores), output_sum
# augmenter
def hello_world_augmenter(text):
    text = ['Hello'] + text.split() + ['World']
    text = ' '.join(text)
    return text

processor = processor.process('bieo', processors=[hello_world_augmenter]).postprocess()

df_dict = processor.get_tts(test_size=0.3)
df_train = df_dict['train'][['text', 'labels']]
df_test = df_dict['test'][['text', 'labels']]

df_label_map = LABELS_MAP_DICT['TUDarmstadt']['bieo']

max_length = 1024

# datasets
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-base', add_prefix_space=True)
model = AutoModel.from_pretrained('google/bigbird-roberta-base')
optimizer = Adam(model.parameters())

trainset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length, strategy = "standard_bieo"
)
testset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length, strategy = "standard_bieo", is_train=False
)

""" strategies: 'standard_io',
                'wordLevel_io', # consider deprecating wordLevel_io as it is the same as standard!
                'standard_bio',
                'wordLevel_bio',
                'standard_bixo',
                'standard_bieo',
                'wordLevel_bieo'"""

train_loader = DataLoader(trainset)
test_loader = DataLoader(testset)

# sample training script (very simplistic, see run.py in cluster/cluster_setup/job_files for a full-fledged one)
clsfr = EDUPredictor()
epochs = 1
for epoch in tqdm(enumerate(range(epochs)), total=epochs, desc='Epochs'):
    model.train()
    train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc='Batches')
    for i, (inputs, targets) in train_loader_tqdm:
        #print(inputs['input_ids'].size())
        #print(inputs['input_ids'])
        #print(targets.size())
        #print(targets)
        optimizer.zero_grad()

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        #print(outputs[0].size())
        out,tag = clsfr(outputs[0])
        #print("out.size(): ",out.size())
        #print("tag.size(): ",tag.size())
        # backward pass
        for i in range(len(targets[0])):
            if targets[0][i] == -100:
                targets[0][i] = 10
        loss = clsfr.crf(tag,targets, reduction = 'token_mean')
        loss.backward()
        optimizer.step()
        train_loader_tqdm.set_postfix ({f"loss: ":{loss.item()}})
# run inference
df_metrics, df_scores = inference(model, test_loader)
