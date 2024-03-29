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
input_size = length
embedding_size = 768
epochs = 60
lrlo = 1e-5
lrcls = 1e-6
window_size = 5
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.4
beta = 0.5
gamma = 0.1
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
special_token = "[EDU]"

# Add the special token to the tokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
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

class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, y_hat, labels):
        mse_loss = nn.MSELoss()(y_hat, labels)
        
        # Compute cosine similarity loss
        similarity_loss = 1 - nn.functional.cosine_similarity(y, y_hat).mean()
        
        # Compute margin ranking loss
        margin_ranking_loss = 0
        N = y.size(0)
        for i in range(N):
            for j in range(i+1, N):
                r_ij = 1 if labels[i] > labels[j] else -1 if labels[i] < labels[j] else torch.sign(y[i] - y[j])
                margin_ranking_loss += torch.max(0, -r_ij * (y[i] - y[j]) + 0)
        margin_ranking_loss /= (N * (N - 1)) / 2  # Normalize by the number of pairs
        
        total_loss = self.alpha * mse_loss + self.beta * margin_ranking_loss + self.gamma * similarity_loss
        return total_loss

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


edu_dir = os.path.join(data_dir,"seg-adu")
def mean_encoding(essay_list, essay_id_list, tokenizer):

  print('Encoding essay embeddings:')
  ip_ids = []
  attn_masks = []
  max_len = 0
  show_once = True
  for (default_essay,essay_id) in tqdm(zip(essay_list, essay_id_list), total=len(essay_list)):
    #essay = essay[:512]
    #print(len(essay))
    essay = ""
    if  os.path.exists(os.path.join(edu_dir, str(essay_id) + ".out")):
        with open(os.path.join(edu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                essay += f"{special_token} {line.strip()} "
            if(show_once):
               print(essay)
               show_once = False
    else:
       print(f"couldn't find edus for essay id: {essay_id} \n Looked at path: {os.path.join(edu_dir, str(essay_id) + '.out')}")
       essay = default_essay
       
    if max_len < len(tokenizer.tokenize(essay)):
       max_len = len(tokenizer.tokenize(essay))
    encoded_input = tokenizer(essay, padding="max_length", truncation=True, max_length=length, return_tensors='pt', return_attention_mask=True, add_special_tokens=True)
    #print(encoded_input["input_ids"].size())
    #with torch.no_grad():
    #print(encoded_input.size())
    tokens_id_mat = np.matrix(encoded_input["input_ids"])
    attn_mask_mat = np.matrix(encoded_input["attention_mask"])
    ip_ids.append(np.squeeze(np.asarray(tokens_id_mat)))
    attn_masks.append(np.squeeze(np.asarray(attn_mask_mat)))
  print(max_len)
  return np.array(ip_ids), np.array(attn_masks)

"""import h5py
embeddings_file = os.path.join(data_dir,f'embeddings_l_adu_{length}.pt')
if os.path.exists(embeddings_file):
    h5f = h5py.File(embeddings_file,'r')
    essay_embeddings = h5f[f'embeddings_l_adu_{length}'][:]
    h5f.close()
    print(f"embeddings loaded from {embeddings_file}")
else:"""
ip_ids, attn_masks = mean_encoding(dataset['essay'], dataset["essay_id"], tokenizer)
"""    print("essay_embeddings got from function")
    h5f = h5py.File(embeddings_file, 'w')
    print("h5f variable init")
    h5f.create_dataset(f'embeddings_l_adu_{length}', data=essay_embeddings)
    print("h5f dataset created")
    h5f.close()
    print("h5f closed")"""

print("embeddings done")

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


def get_loader(df, id2emb, ip_ids, attn_masks, shuffle=True):

  # get embeddings from essay_id using id2emb dict
  ip = np.array([ip_ids[id2emb[id]] for id in df['essay_id']])
  attn = np.array([attn_masks[id2emb[id]] for id in df['essay_id']])
  # dataset and dataloader
  data = TensorDataset(torch.from_numpy(ip).long(), torch.from_numpy(attn).float(), torch.from_numpy(np.array(df['scaled_score'])).float())
  loader = DataLoader(data, batch_size=16, shuffle=shuffle, num_workers=0)

  return loader

class LongFo(torch.nn.Module):
  def __init__(self):
    super(LongFo, self).__init__()
    self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
    self.model.resize_token_embeddings(len(tokenizer))     
  def forward(self,ip,mask):
    model_output = self.model(input_ids=ip.to(torch.long),attention_mask=mask)
    return model_output
class MLP(torch.nn.Module):
  def __init__(self, input_size,embedding_size, window_size):
    super(MLP, self).__init__()
    self.window_size = window_size
    self.p = 0.4
    self.lstm1 = nn.LSTM(768, 512, batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(p=self.p)
    self.layers1 = torch.nn.Sequential(
      torch.nn.Linear(512*2, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(96, 1)
    )
    self.lstm2 = nn.LSTM(input_size, 512, batch_first=True, bidirectional=True)
    self.dropout2 = nn.Dropout(p=self.p)
    self.layers2 = torch.nn.Sequential(
      torch.nn.Linear(512*2, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=self.p),
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
        l1out, _ = self.lstm1(x) 
        l1out = self.dropout1(l1out)
        layer_1_out = self.layers1(l1out)
        #print("layer_1_out: ",layer_1_out.size())
        layer_1_out = layer_1_out.squeeze(dim=-1)
        #print("layer_1_out squeezed: ",layer_1_out.size())
        l2out, _ = self.lstm2(layer_1_out) 
        l2out = self.dropout2(l2out)
        layer_2_out = self.layers2(l2out)
        #print("layer_2_out: ",layer_2_out.size())
        return layer_2_out
  
class Ngram_Clsfr(nn.Module):
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

        return h




def training_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, train_loader):

  samples = 0.
  cumulative_loss = 0.

  trans.train() 
  clsfr.train()
  train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc='Batches')
  for step, (inputs, mask, targets) in enumerate(train_loader_tqdm):
 
    #print("before sq ip size:",inputs.size())
    inputs = inputs.squeeze(dim=1).to(device)
    #print("after sq ip size:",inputs.size())
    mask = mask.squeeze(dim=1).to(device)
    targets = targets.reshape(targets.shape[0],1).to(device)

    trans_op = trans(inputs,mask)
    #print("before sq trans op size:",trans_op[0].size())
    #print("after sq trans op size:",trans_op[0].squeeze().size())
    clsfr_op = clsfr(trans_op[0].squeeze())
    loss = cost_function(clsfr_op, targets)

    loss.backward()  
  
    optimizerLo.step()  
 
    optimizerLo.zero_grad()
    optimizerCls.step()  
 
    optimizerCls.zero_grad()

    samples += inputs.shape[0]
    cumulative_loss += loss.item()

  return cumulative_loss/samples


def test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, test_loader):

  samples = 0.
  cumulative_loss = 0.
  preds = []

  trans.eval() 
  clsfr.eval()
  with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc='Test Batches')
    for step, (inputs, mask, targets) in enumerate(test_loader_tqdm):

      inputs = inputs.squeeze(dim=1).to(device)
      mask = mask.squeeze(dim=1).to(device)
      targets = targets.reshape(targets.shape[0],1).to(device)

      trans_op = trans(inputs,mask)
      clsfr_op = clsfr(trans_op[0].squeeze())
      loss = cost_function(clsfr_op, targets)

      loss = cost_function(clsfr_op, targets)

      samples += inputs.shape[0]
      cumulative_loss += loss.item()
      for out in clsfr_op:
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
best_lo_path = os.path.join(data_dir,'long_best.pth')
best_clsfr_path = os.path.join(data_dir,'clsfr_best.pth')
# copy of dataset with scaled scores computed using the whole dataset
print("copy of scaled_dataset begin")
scaled_dataset = get_scaled_dataset(dataset)
print("copy of scaled_dataset end")
gc.collect()
print("gc collected #1")
data = ""
for n, (train, test) in enumerate(kf.split(dataset)):
    if(n==0):
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
        train_loader = get_loader(train_df, id2emb, ip_ids, attn_masks, shuffle=True)
        print("train_loader")
        test_loader = get_loader(test_df, id2emb, ip_ids, attn_masks, shuffle=False)
        print("test_loader")
        # model
        print('------------------------------------------------------------------')
        print(f"\t\t\tTraining model: {n+1}")
        print('------------------------------------------------------------------')
        trans = nn.DataParallel(LongFo()).to(device)
        clsfr = nn.DataParallel(MLP(input_size, embedding_size, window_size)).to(device)
        #model = Ngram_Clsfr().to(device)
        # loss and optimizer
        cost_function = CombinedLoss(alpha, beta, gamma)
        optimizerLo = torch.optim.Adam(trans.parameters(), lr=lrlo)
        optimizerCls = torch.optim.Adam(clsfr.parameters(), lr=lrcls)
        torch.cuda.empty_cache()
        # training
        #train_loss, train_preds = test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, train_loader)
        #test_loss, test_preds = test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, test_loader)
        #print('Before training:\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))
        best_loss = 1.0
        epoch_tqdm = tqdm(range(epochs), total=epochs, desc='Epochs')
        for epoch in epoch_tqdm:
            train_loss = training_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, train_loader)
            test_loss, test_preds = test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, test_loader)
            if test_loss < best_loss:
                torch.save(clsfr.state_dict(), best_clsfr_path)
                torch.save(trans.state_dict(), best_lo_path)
                print("Saving model")
                best_loss = test_loss

            epoch_tqdm.set_postfix ({f"Epoch: {epoch+1} \t\t Train Loss: {train_loss:.5f} Test Loss: {test_loss:.5f} \n":  test_loss})

        clsfr.load_state_dict(torch.load(best_clsfr_path))
        trans.load_state_dict(torch.load(best_lo_path))
        train_loss, train_preds = test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, train_loader)
        test_loss, test_preds = test_step(trans, clsfr, cost_function, optimizerLo, optimizerCls, test_loader)
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
