from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor
from argminer.evaluation import inference
from argminer.config import LABELS_MAP_DICT
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel, AutoConfig, RobertaForSequenceClassification, RobertaForTokenClassification
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

def hello_world_augmenter(text):
    text = ['Hello'] + text.split() + ['World']
    text = ' '.join(text)
    return text

processor = processor.process('bieo', processors=[hello_world_augmenter]).postprocess()

df_dict = processor.get_tts(test_size=0.3)
df_train = df_dict['train'][['text', 'labels']]
df_test = df_dict['test'][['text', 'labels']]

df_label_map = LABELS_MAP_DICT['TUDarmstadt']['bieo']

max_length = 512

# datasets
id2label =  {-100: "Ignore", 0: 'O', 1: 'B-MajorClaim', 2: 'I-MajorClaim', 3: 'E-MajorClaim', 4: 'B-Claim', 5: 'I-Claim', 6: 'E-Claim', 7: 'B-Premise', 8: 'I-Premise', 9: 'E-Premise'}
label2id = {'O': 0, 'B-MajorClaim': 1, 'I-MajorClaim': 2, 'E-MajorClaim': 3, 'B-Claim': 4, 'I-Claim': 5, 'E-Claim': 6, 'B-Premise': 7, 'I-Premise': 8, 'E-Premise': 9, "Ignore": -100}
max_position_embeddings = max_length
model = RobertaForTokenClassification.from_pretrained('google/bigbird-roberta-base', id2label = id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-base', add_prefix_space=True)



print(model.config)
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
epochs_data = []
epochs = 100

for epoch in tqdm(enumerate(range(epochs)), total=epochs, desc='Epochs'):
    model.train()
    train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc='Batches')
    for i, (inputs, targets) in train_loader_tqdm:
        #print(inputs['input_ids'].size())
        #print(inputs['input_ids'])
        #print(targets.size())
        #print(targets)
        #print(sorted(torch.unique(targets)))
        optimizer.zero_grad()

        loss, outputs = model(
            labels=targets,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=False
        )
        loss.backward()
        optimizer.step()
        train_loader_tqdm.set_postfix ({f"loss: ":{loss.item()}})
    df_metrics, df_scores = inference(model, test_loader)
    class_avg_f1_scores = df_scores.groupby('class')['f1'].mean()
    class_avg_f1_dict = class_avg_f1_scores.to_dict()
    print(class_avg_f1_dict)
    #print(type(class_avg_f1_dict))
    #print(class_avg_f1_dict.keys())
    epochs_data.append(class_avg_f1_dict)

print("Class-specific Average F1 Scores over Epochs:")
for epoch, scores in enumerate(epochs_data, 1):
    print("Epoch", epoch, ":", scores)
