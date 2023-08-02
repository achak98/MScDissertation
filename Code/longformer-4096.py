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
# set device
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

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
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

#tokenizer = AutoTokenizer.from_pretrained("roberta-base")

length_dict = {}

for essay_set in range(1, 9):
    length_dict[essay_set] = [
        len(tokenizer.tokenize(essay))
        for essay in dataset[dataset["essay_set"] == essay_set]["essay"]
    ]


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

roberta = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)



def mean_encoding(essay_list, model, tokenizer):

  print('Encoding essay embeddings:')
  embeddings = []
  for essay in tqdm(essay_list):
    #essay = essay[:512]
    #print(len(essay))
    encoded_input = tokenizer(essay, padding="max_length", max_length=2048, return_tensors='pt', return_attention_mask=True).to(device)
    #print(encoded_input["input_ids"].size())
    with torch.no_grad():
      model_output = model(**encoded_input)
    #print(model_output[0].squeeze().size())
    tokens_embeddings = np.matrix(model_output[0].squeeze().cpu())
    embeddings.append(np.squeeze(np.asarray(tokens_embeddings)))
  return np.array(embeddings)

import h5py
embeddings_file = os.path.join(data_dir,'embeddings_l_2048.pt')
if os.path.exists(embeddings_file):
    h5f = h5py.File(embeddings_file,'r')
    essay_embeddings = h5f['embeddings_l_2048'][:]
    h5f.close()
    print(f"embeddings loaded from {embeddings_file}")
else:
    essay_embeddings = mean_encoding(dataset['essay'], roberta, tokenizer)
    h5f = h5py.File(embeddings_file, 'w')
    h5f.create_dataset('embeddings_l_2048', data=essay_embeddings)
    h5f.close()

print("embeddings done")


def get_loader(df, id2emb, essay_embeddings, shuffle=True):

  # get embeddings from essay_id using id2emb dict
  embeddings = np.array([essay_embeddings[id2emb[id]] for id in df['essay_id']])

  # dataset and dataloader
  data = TensorDataset(torch.from_numpy(embeddings).float(), torch.from_numpy(np.array(df['scaled_score'])).float())
  loader = DataLoader(data, batch_size=128, shuffle=shuffle, num_workers=0)

  return loader



class MLP(torch.nn.Module):
  
  def __init__(self, input_size,embedding_size, window_size):
    super(MLP, self).__init__()
    self.window_size = window_size
    self.layers1 = torch.nn.Sequential(
      torch.nn.Linear(768, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(96, 1)
    )
    self.lstm1 = nn.LSTM(input_size, input_size, num_layers=1, bidirectional=True)
    self.dropout1 = nn.Dropout(0.3) 
    self.fc1 = nn.Linear(input_size*2,input_size)
    self.dropout2 = nn.Dropout(0.3)
    self.attention_weights = nn.Linear(3, 1)
    self.dropout3 = nn.Dropout(0.3) 
    self.lstm2 = nn.LSTM(input_size, input_size, num_layers=1, bidirectional=True)
    self.dropout4 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(input_size*2,input_size) 
    self.dropout5 = nn.Dropout(0.3)
    self.layers2 = torch.nn.Sequential(
      torch.nn.Linear(input_size, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(96, 1)
    ) 

  def similarity(self, hi, hj):
        # Concatenate the hidden representations
        """print("hi: ",hi.size())
        print("hj: ",hj.size())
        print("hi * hj: ",(hi * hj).size())"""
        h_concat = torch.cat([hi, hj, hi * hj], dim=-1)
        #print("h_concat: ",h_concat.size())
        attn_weights = self.attention_weights(h_concat)
        #print("attn_weights: ",attn_weights.size())
        return attn_weights
    
  def forward(self, x):
        #print("x: ",x.size())
        layer_1_out = self.layers1(x)
        #print("layer_1_out: ",layer_1_out.size())
        layer_1_out = layer_1_out.squeeze()
        #print("layer_1_out squeezed: ",layer_1_out.size())
        
        layer_2_out = self.layers2(layer_1_out)
        #print("layer_2_out: ",layer_2_out.size())
        return layer_2_out


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
input_size = 2048
embedding_size = 768
epochs = 15
lr = 3e-4
window_size = 5
# cross-validation folds
kf = KFold(n_splits=10, random_state=2022, shuffle=True)
print("after kfold init")
# dicts with train_df, test_df and predictions for each model
train_df_dict = {}
test_df_dict = {}
preds_dict = {}

# copy of dataset with scaled scores computed using the whole dataset
print("copy of scaled_dataset begin")
scaled_dataset = get_scaled_dataset(dataset)
print("copy of scaled_dataset end")
gc.collect()
print("gc collected #1")
#for n, (train, test) in enumerate(kf.split(dataset)):
n = 0
train, test =kf.split(dataset)[n]
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
model = MLP(input_size, embedding_size, window_size).to(device)

# loss and optimizer
cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training
train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
print('Before training:\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))

epoch_tqdm = tqdm(range(epochs), total=epochs, desc='Epochs')
for epoch in epoch_tqdm:
train_loss = training_step(model, cost_function, optimizer, train_loader)
test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
epoch_tqdm.set_postfix ({f"Epoch: {epoch+1} \t\t Train Loss: {train_loss:.5f} Test Loss: {test_loss:.5f} \n":  test_loss})


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
data = ""
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

  


