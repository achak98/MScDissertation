import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score as kappa
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import LongformerModel, AutoTokenizer
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch.nn as nn
import os
import gc

import warnings
warnings.filterwarnings("ignore")

length = 1792
alpha = 0.9
beta = 0.1
gamma = 0.0
input_size = length
embedding_size = 768
epochs = 60
lr = 3e-4
window_size = 5
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "./../Data/ASAP-AES/"
# Original kaggle training set
kaggle_dataset = pd.read_csv(
    os.path.join(data_dir,"training_set_rel3.tsv"), sep="\t", encoding="ISO-8859-1"
)
# Smaller training set used for this project
dataset = pd.DataFrame(
    {
        "essay_id": kaggle_dataset["essay_id"],
        "essay_set": kaggle_dataset["essay_set"],
        "essay": kaggle_dataset["essay"],
        "rater1": kaggle_dataset["rater1_domain1"],
        "rater2": kaggle_dataset["rater2_domain1"],
        "score": kaggle_dataset["domain1_score"],
    }
)

#transformer_architecture = 'roberta-base' 
#config = AutoConfig.from_pretrained(transformer_architecture, output_hidden_states=True)
#config.max_position_embeddings = 512
special_token_adu = "[ADU]"
special_token_prompt = "[PROMPT]"
# Add the special token to the tokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer.add_special_tokens({"additional_special_tokens": [special_token_adu, special_token_prompt]})
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")

length_dict = {}

for essay_set in range(1, 9):
    length_dict[essay_set] = [
        len(tokenizer.tokenize(essay))
        for essay in dataset[dataset["essay_set"] == essay_set]["essay"]
    ]
    print(max(length_dict[essay_set]))
def get_scaled_dataset(dataset):
    scaler = StandardScaler()
    scaled = []
    for essay_set in range(1, 9):
        score = dataset[dataset["essay_set"] == essay_set]["score"].to_frame()
        s = scaler.fit_transform(score).reshape(-1)
        scaled = np.append(scaled, s)

    scaled_dataset = dataset.copy()
    scaled_dataset["scaled_score"] = scaled

    return scaled_dataset


def get_id2emb(ids):
    id2emb = {}
    for n, id in enumerate(ids.to_list()):
        id2emb[id] = n
    print("Essay ids to emebeddings dictionary created.")

    return id2emb


id2emb = get_id2emb(dataset["essay_id"])
prompts_dict = {
    1:"Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.",
    2:"Censorship in the Libraries Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.",
    3:"ROUGH ROAD AHEAD: Do Not Exceed Posted Speed Limit by Joe Kurmaskie Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.",
    4:"Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.",
    5:"Narciso Rodriguez from Home: The Blueprints of Our Lives Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.",
    6:"The Mooring Mast by Marcia Amidon Lüsted Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.",
    7:"Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.",
    8:"We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part."
}
roberta = LongformerModel.from_pretrained("allenai/longformer-base-4096")
roberta.resize_token_embeddings(len(tokenizer))
roberta = nn.DataParallel(roberta).to(device)
edu_dir = os.path.join(data_dir,"seg-adu")
def mean_encoding(essay_list, essay_id_list, essay_set_list, model, tokenizer):

  print('Encoding essay embeddings:')
  embeddings = []
  max_len = 0
  show_once = True
  for (default_essay,essay_id,essay_set) in tqdm(zip(essay_list, essay_id_list, essay_set_list), total=len(essay_list)):
    #essay = essay[:512]
    #print(len(essay))
    essay = f"{special_token_prompt} {prompts_dict[int(essay_set)]} "
    if  os.path.exists(os.path.join(edu_dir, str(essay_id) + ".out")):
        with open(os.path.join(edu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                essay += f"{special_token_adu} {line.strip()} "
            if(show_once):
               
               print(essay)
               show_once = False
    else:
       print(f"couldn't find edus for essay id: {essay_id} \n Looked at path: {os.path.join(edu_dir, str(essay_id) + '.out')}")
       essay = default_essay
       
    if max_len < len(tokenizer.tokenize(essay)):
       max_len = len(tokenizer.tokenize(essay))
    encoded_input = tokenizer(essay, padding="max_length", truncation=True, max_length=length, return_tensors='pt', return_attention_mask=True, add_special_tokens=True).to(device)
    #print(encoded_input["input_ids"].size())
    with torch.no_grad():
      model_output = model(**encoded_input)
    tokens_embeddings = np.matrix(model_output[0].squeeze().cpu())
    embeddings.append(np.squeeze(np.asarray(tokens_embeddings)))
  print(max_len)
  return np.array(embeddings)

import h5py
embeddings_file = os.path.join(data_dir,f'embeddings_l_adu_prompt_{length}.pt')
if os.path.exists(embeddings_file):
    h5f = h5py.File(embeddings_file,'r')
    essay_embeddings = h5f[f'embeddings_l_adu_prompt_{length}'][:]
    h5f.close()
    print(f"embeddings loaded from {embeddings_file}")
else:
    essay_embeddings = mean_encoding(dataset['essay'], dataset["essay_id"], dataset["essay_set"], roberta, tokenizer)
    print("essay_embeddings got from function")
    h5f = h5py.File(embeddings_file, 'w')
    print("h5f variable init")
    h5f.create_dataset(f'embeddings_l_adu_prompt_{length}', data=essay_embeddings)
    print("h5f dataset created")
    h5f.close()
    print("h5f closed")

print("embeddings done")


def get_loader(df, id2emb, essay_embeddings, shuffle=True):

  # get embeddings from essay_id using id2emb dict
  embeddings = np.array([essay_embeddings[id2emb[id]] for id in df['essay_id']])

  # dataset and dataloader
  data = TensorDataset(torch.from_numpy(embeddings).float(), torch.from_numpy(np.array(df['scaled_score'])).float())
  loader = DataLoader(data, batch_size=64, shuffle=shuffle, num_workers=0)

  return loader


class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, labels):
        mse_loss = nn.MSELoss()(y, labels)
        
        # Compute cosine similarity loss
        similarity_loss = 1 - nn.functional.cosine_similarity(y, labels).mean()
        
        # Compute margin ranking loss
        margin_ranking_loss = 0
        N = y.size(0)
        for i in range(N):
            for j in range(i+1, N):
                r_ij = 1 if labels[i] > labels[j] else -1 if labels[i] < labels[j] else torch.sign(y[i] - y[j])
                margin_ranking_loss += torch.max(torch.tensor(0), -r_ij * (y[i] - y[j]) + torch.tensor(0))
        margin_ranking_loss /= (N * (N - 1)) / 2  # Normalize by the number of pairs
        
        total_loss = self.alpha * mse_loss + self.beta * margin_ranking_loss + self.gamma * similarity_loss
        return total_loss
  
"""class Ngram_Clsfr(nn.Module):
    def __init__(self):
        super(Ngram_Clsfr, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride = 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru1 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.4)

        self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3, stride = 3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.gru2 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4, stride = 4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.gru3 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(p=0.4)

        self.fc = nn.Linear(128*2*3,1)

    def forward(self, x):
        print(f"x: {x.size()}")
        x=x.permute(0,2,1)
        print(f"x: {x.size()}")
        x1 = self.conv1(x)
        print(f"x1: {x1.size()}")
        x1 = self.pool1(x1)
        print(f"x1: {x1.size()}")
        x1=x1.permute(0,2,1)
        print(f"x1: {x1.size()}")
        h1, _ = self.gru1(x1) #x1 should be batch size, sequence length, input length
        print(f"h1: {h1.size()}")
        h1 = torch.cat((h1[0, :, :], h1[1, :, :]), dim=1)
        print(f"h1: {h1.size()}")
        h1 = self.dropout1(h1)
        print(f"h1: {h1.size()}")

        x2 = self.conv2(x)
        x2 = self.pool2(x2)
        x2=x2.permute(0,2,1)
        h2, _ = self.gru2(x2)
        h2 = torch.cat((h2[0, :, :], h2[1, :, :]), dim=1)
        h2 = self.dropout1(h2)

        x3 = self.conv3(x)
        x3 = self.pool3(x3)
        x3=x3.permute(0,2,1)
        h3, _ = self.gru3(x3)
        h3 = torch.cat((h3[0, :, :], h3[1, :, :]), dim=1)
        h3 = self.dropout1(h3)

        h = torch.cat((h1, h2, h3), dim=1)
        print(f"h: {h.size()}")
        h = self.fc(h)
        print(f"h: {h.size()}")
        h = h.squeeze()
        print(f"h: {h.size()}")

        return h"""
class Ngram_Clsfr(nn.Module):
    def __init__(self):
        super(Ngram_Clsfr, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        super(Ngram_Clsfr, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride = 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru1 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.4)

        self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3, stride = 3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.gru2 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4, stride = 4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.gru3 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(p=0.4)

        self.fc = nn.Linear(128*2*3,1)

    def forward(self, x):
        print(f"x: {x.size()}")
        x=x.permute(0,2,1)
        print(f"x after permute: {x.size()}")
        x1 = self.conv1(x)
        print(f"x1 conv: {x1.size()}")
        x1 = self.pool1(x1)
        print(f"x1 pool: {x1.size()}")
        x1=x1.permute(0,2,1)
        print(f"x1 after permute: {x1.size()}")
        h1, _ = self.gru1(x1) #x1 should be batch size, sequence length, input length
        print(f"h1 lstm: {h1.size()}")
        #h1 = torch.cat((h1[0, :, :], h1[1, :, :]), dim=1)
        #print(f"h1: {h1.size()}")
        h1 = self.dropout1(h1)
        print(f"h1 do: {h1.size()}")

        x2 = self.conv2(x)
        print(f"x2 conv: {x2.size()}")
        x2 = self.pool2(x2)
        print(f"x2 pool: {x2.size()}")
        x2=x2.permute(0,2,1)
        print(f"x2 after permute: {x2.size()}")
        h2, _ = self.gru2(x2)
        print(f"h2 lstm: {h2.size()}")
        #h2 = torch.cat((h2[0, :, :], h2[1, :, :]), dim=1)
        h2 = self.dropout1(h2)
        print(f"h2 do: {h2.size()}")

        x3 = self.conv3(x)
        print(f"x3 conv: {x3.size()}")
        x3 = self.pool3(x3)
        print(f"x3 pool: {x3.size()}")
        x3=x3.permute(0,2,1)
        print(f"x3 after permute: {x3.size()}")
        h3, _ = self.gru3(x3)
        print(f"h3 lstm: {h3.size()}")
        #h3 = torch.cat((h3[0, :, :], h3[1, :, :]), dim=1)
        h3 = self.dropout1(h3)
        print(f"h3 do: {h3.size()}")

        h = torch.cat((h1, h2, h3), dim=1)
        print(f"h after cat: {h.size()}")
        h = self.fc(h)
        print(f"h after fc: {h.size()}")
        h = h.squeeze()
        print(f"h after squeeze: {h.size()}")

        return h




def training_step(model, cost_function, optimizer, train_loader):

  samples = 0.
  cumulative_loss = 0.

  model.train() 
  train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc='Batches')
  for step, (inputs, targets) in enumerate(train_loader_tqdm):

    inputs = inputs.squeeze(dim=1).to(device)
    targets = targets.reshape(targets.shape[0],1).to(device)

    outputs = model(inputs)

    loss = cost_function(outputs, targets)

    loss.backward()  
  
    optimizer.step()  
 
    optimizer.zero_grad()

    samples += inputs.shape[0]
    cumulative_loss += loss.item()

  return cumulative_loss/samples


def test_step(model, cost_function, optimizer, test_loader):

  samples = 0.
  cumulative_loss = 0.
  preds = []

  model.eval() 

  with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc='Test Batches')
    for step, (inputs, targets) in enumerate(test_loader_tqdm):

      inputs = inputs.squeeze(dim=1).to(device)
      targets = targets.reshape(targets.shape[0],1).to(device)

      outputs = model(inputs)

      loss = cost_function(outputs, targets)

      samples += inputs.shape[0]
      cumulative_loss += loss.item()
      for out in outputs:
        preds.append(float(out))

  return cumulative_loss/samples, preds

def get_results_df(train_df, test_df, model_preds):
    # create new results df with model scaled preds
    preds_df = pd.DataFrame(model_preds)
    results_df = (
        test_df.reset_index(drop=True)
        .join(preds_df)
        .rename(columns={0: "scaled_pred"})
        .sort_values(by="essay_set")
        .reset_index(drop=True)
    )

    # move score to last colum
    s_df = results_df.pop("score")
    results_df["score"] = s_df

    # scale back to original range by essay set
    preds = pd.Series(dtype="float64")
    for essay_set in range(1, 9):
        scaler = StandardScaler()
        score_df = train_df[train_df["essay_set"] == essay_set]["score"].to_frame()
        scaler.fit(score_df)
        scaled_preds = results_df.loc[
            results_df["essay_set"] == essay_set, "scaled_pred"
        ].to_frame()
        preds_rescaled = scaler.inverse_transform(scaled_preds).round(0).astype("int")
        preds = preds.append(
            pd.Series(np.squeeze(np.asarray(preds_rescaled))), ignore_index=True
        )

    # append to results df
    results_df["pred"] = preds

    return results_df

print("before hypparams")
# hyper-parameters

# cross-validation folds
kf = KFold(n_splits=10, random_state=2022, shuffle=True)
print("after kfold init")
# dicts with train_df, test_df and predictions for each model
train_df_dict = {}
test_df_dict = {}
preds_dict = {}
best_model_path = os.path.join(data_dir,'long_best_all.pth')
# copy of dataset with scaled scores computed using the whole dataset
print("copy of scaled_dataset begin")
scaled_dataset = get_scaled_dataset(dataset)
print("copy of scaled_dataset end")
gc.collect()
print("gc collected #1")
data = ""
for n, (train, test) in enumerate(kf.split(dataset)):

    print("train test split done")
    gc.collect()
    print("gc collected #2")
    # train, test splits 
    # scaled scores in train_df are computed only using training data
    train_df = dataset.iloc[train]
    print("train_df #1")
    train_df = get_scaled_dataset(train_df)
    print("train_df #2")
    test_df = scaled_dataset.iloc[test]
    print("train_df #1")
    # dataloaders
    train_loader = get_loader(train_df, id2emb, essay_embeddings, shuffle=True)
    print("train_loader")
    test_loader = get_loader(test_df, id2emb, essay_embeddings, shuffle=False)
    print("test_loader")
    # model
    print('------------------------------------------------------------------')
    print(f"\t\t\tTraining model: {n+1}")
    print('------------------------------------------------------------------')
    #model = nn.DataParallel(MLP(input_size, embedding_size, window_size)).to(device)
    model = nn.DataParallel(Ngram_Clsfr()).to(device)
    # loss and optimizer
    cost_function = CombinedLoss(alpha, beta, gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
    test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
    print('Before training:\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))
    best_loss = 1.0
    epoch_tqdm = tqdm(range(epochs), total=epochs, desc='Epochs')
    for epoch in epoch_tqdm:
        train_loss = training_step(model, cost_function, optimizer, train_loader)
        test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
        if test_loss < best_loss:
            torch.save(model.state_dict(), best_model_path)
            print("Saving model")
            best_loss = test_loss

        epoch_tqdm.set_postfix ({f"Test Loss: {test_loss:.5f} Train Loss: {train_loss:.5f} for Epoch: ":  {epoch+1}})

    model.load_state_dict(torch.load(best_model_path))
    train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
    test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
    print('After training:\t\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))

    print("getting results df")
    results_df = get_results_df(train_df, test_df, test_preds)
    print("got results df")
    kappas_by_set = []
    for essay_set in range(1, 9):
        kappas_by_set.append(
            kappa(
                results_df.loc[results_df["essay_set"] == essay_set, "score"],
                results_df.loc[results_df["essay_set"] == essay_set, "pred"],
                weights="quadratic",
            )
        )
        print(f"got kappa for essay set {essay_set}")
    id = n + 1
    
    print("--------------------------------------")
    print(f"\tResults for model: {id}")
    print("--------------------------------------")
    data += "\n--------------------------------------"
    data += f"\n\tResults for model: {id}"
    data += "\n--------------------------------------"
    for essay_set in range(8):
        data += "\nKappa for essay set {:}:\t\t{:.4f}".format(
            essay_set + 1, kappas_by_set[essay_set]
        )
        print(
            "Kappa for essay set {:}:\t\t{:.4f}".format(
                essay_set + 1, kappas_by_set[essay_set]
            )
        )
    data += "\nmean QWK:\t\t\t{:.4f}".format(np.mean(kappas_by_set))
    print("mean QWK:\t\t\t{:.4f}".format(np.mean(kappas_by_set)))




def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


# Example usage:
save_directory = "./../Data/results/longformer-edu"
check_and_create_directory(save_directory)

file = open(os.path.join(save_directory, f"qwk.txt"), "w")
file.write(data)
file.close()
