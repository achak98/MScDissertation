import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
essay_set = 1
#length1 = 128
#length2 = 1536
length_context = 3
embedding_size = 768
length_emb = 1201 #1201 
len_tot = length_emb + length_context
alpha = 0.9
beta = 0.1
gamma = 0.0
input_size = length_emb

epochs = 150
lr = 3e-5
window_size = 5
dor = 0.4

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
dataset = dataset[dataset["essay_set"] == essay_set]
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


length_dict[essay_set] = [
    len(tokenizer.tokenize(essay))
    for essay in dataset[dataset["essay_set"] == essay_set]["essay"]
]
print(max(length_dict[essay_set]))

def get_scaled_dataset(dataset):
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = []

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
edu_dir = os.path.join(data_dir,"seg-edu")
adu_dir = os.path.join(data_dir,"seg-adu")
def count_words(text):
    words = text.split()
    return len(words)

def mean_encoding(essay_list, essay_id_list, essay_set_list, model, tokenizer):

  print('Encoding essay embeddings:')
  mat_embeddings = []
  mat_len_context = []
  max_len = 0
  show_once = True
  for (default_essay,essay_id,essay_set) in tqdm(zip(essay_list, essay_id_list, essay_set_list), total=len(essay_list)):
    #essay = essay[:512]
    #print(len(essay))
    #prompt = prompts_dict[int(essay_set)]
    #encoded_p = tokenizer(prompt, padding="max_length", truncation=True, max_length=length2, return_tensors='pt', return_attention_mask=True, add_special_tokens=True).to(device)
    #print(encoded_input["input_ids"].size())
    #with torch.no_grad():
    #  model_output = model(**encoded_p)
    #  prompt_embed = model_output[0].squeeze().cpu()
    essay = f"" #{special_token_prompt} {prompt}"
    no_of_adus = 0
    no_of_edus = 0
    if  os.path.exists(os.path.join(adu_dir, str(essay_id) + ".out")):
        with open(os.path.join(adu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                essay += f"{special_token_adu} {line.strip()} "
                no_of_adus+=1
            if(show_once):
               print(f"adus read from dir: {os.path.join(adu_dir, str(essay_id) + '.out')}")
               print(essay)
    else:
       print(f"couldn't find edus for essay id: {essay_id} \n Looked at path: {os.path.join(edu_dir, str(essay_id) + '.out')}")
       essay = default_essay
    if  os.path.exists(os.path.join(edu_dir, str(essay_id) + ".out")):
        with open(os.path.join(edu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                no_of_edus+=1
            if show_once:
                print(f"edus read from dir: {os.path.join(edu_dir, str(essay_id) + '.out')}")
    if max_len < len(tokenizer.tokenize(essay)):
       max_len = len(tokenizer.tokenize(essay))
    encoded_input = tokenizer(essay, padding="max_length", truncation=True, max_length=length_emb, return_tensors='pt', return_attention_mask=True, add_special_tokens=True).to(device)
    with torch.no_grad():
      model_output = model(**encoded_input)
      embeddings = model_output[0].squeeze().cpu()
    wc = torch.tensor(count_words(default_essay)).unsqueeze(0)
    educ = torch.tensor(no_of_edus).unsqueeze(0)
    aduc = torch.tensor(no_of_adus).unsqueeze(0)
    #spacer1 = torch.tensor(-1).unsqueeze(0)
    #spacer2 =torch.full((1,768), -1)
    #combined_embed = torch.cat((prompt_embed,spacer2,embeddings), dim=0)
    comb_len_context = torch.cat((wc,educ,aduc), dim=0)
    #print(comb_len_context)
    if show_once:
        print("len context: ",comb_len_context)
        print("combined_embed: ",embeddings.size())
        print("comb_len_context: ",comb_len_context.size())
        show_once = False

    tokens_embeddings = np.matrix(embeddings)
    len_context = np.matrix(comb_len_context)
    mat_embeddings.append(np.squeeze(np.asarray(tokens_embeddings)))
    mat_len_context.append(np.squeeze(np.asarray(len_context)))
  print(max_len)
  return np.array(mat_embeddings), np.array(mat_len_context)

import h5py
embeddings_file = os.path.join(data_dir,f'embeddings_l_len_adu_prompt_{length_emb}{essay_set}.pt')
context_file = os.path.join(data_dir,f'context_l_len_adu_prompt_{length_emb}{essay_set}.pt')
if os.path.exists(embeddings_file):
    h5f = h5py.File(embeddings_file,'r')
    essay_embeddings = h5f[f'embeddings_l_len_adu_prompt_{length_emb}{essay_set}'][:]
    context_embeddings = h5f[f'context_l_len_adu_prompt_{length_emb}{essay_set}'][:]
    h5f.close()
    print(f"embeddings loaded from {embeddings_file}{essay_set}")
else:
    essay_embeddings, context_embeddings = mean_encoding(dataset['essay'], dataset["essay_id"], dataset["essay_set"], roberta, tokenizer)
    print("essay_embeddings got from function")
    h5f = h5py.File(embeddings_file, 'w')
    print("h5f variable init")
    h5f.create_dataset(f'embeddings_l_len_adu_prompt_{length_emb}{essay_set}', data=essay_embeddings)
    h5f.create_dataset(f'context_l_len_adu_prompt_{length_emb}{essay_set}', data=context_embeddings)
    print("h5f dataset created")
    h5f.close()
    print("h5f closed")

print("embeddings done")


def get_loader(df, id2emb, essay_embeddings, context_embeddings, shuffle=True):

  # get embeddings from essay_id using id2emb dict
  embeddings = np.array([essay_embeddings[id2emb[id]] for id in df['essay_id']])
  context = np.array([context_embeddings[id2emb[id]] for id in df['essay_id']])
  
  # dataset and dataloader
  data = TensorDataset(torch.from_numpy(embeddings).float(), torch.from_numpy(context).float(), torch.from_numpy(np.array(df['scaled_score'])).float())
  loader = DataLoader(data, batch_size=128, shuffle=shuffle, num_workers=0)

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

class MLP(torch.nn.Module):
  
  def __init__(self, input_size,embedding_size, window_size):
    super(MLP, self).__init__()
    self.window_size = window_size
    self.p = dor
    self.lstm1 = nn.LSTM(embedding_size, 512, batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(p=self.p)
    self.layers1 = torch.nn.Sequential(
      torch.nn.Linear(512*2, 256),
      torch.nn.Tanh(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(256, 96),
      nn.Tanh(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(96, 1)
    )
    self.fcs = torch.nn.Sequential(
       torch.nn.Linear(len_tot, len_tot),
       nn.Tanh(),
       torch.nn.Dropout(p=self.p)
    )
    self.lstm2 = nn.LSTM(len_tot, 512, batch_first=True, num_layers=1, bidirectional=True)
    self.dropout2 = nn.Dropout(p=self.p)
    self.layers2 = torch.nn.Sequential(
      torch.nn.Linear(512*2, 256),
      torch.nn.Tanh(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(256, 96),
      torch.nn.Tanh(),
      torch.nn.Dropout(p=self.p),
      torch.nn.Linear(96, 1)
    ) 
    
  def forward(self, x, count_context):
        #print("x: ",x.size())
        l1out, _ = self.lstm1(x)
        #l1out = torch.relu(l1out)
        l1out = self.dropout1(l1out)
        layer_1_out = self.layers1(l1out)
        
        layer_1_out = layer_1_out.squeeze()
        added_context = torch.cat((count_context, layer_1_out), dim=1)
        #print(f"layer1_out squeezed: {layer_1_out.size()} || added_context: {added_context.size()}") 
        #interim = self.fcs(added_context)
        #print(f"interim: {interim.size()}")
        l2out, _ = self.lstm2(added_context)
        #l2out = torch.relu(l2out)
        l2out = self.dropout2(l2out)
        layer_2_out = self.layers2(l2out)
        
        #print("layer_2_out: ",layer_2_out.size())
        return layer_2_out
  

def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


# Example usage:
save_directory = "./../Data/results/longformer-len"
check_and_create_directory(save_directory)


def training_step(model, cost_function, optimizer, train_loader):

  samples = 0.
  cumulative_loss = 0.

  model.train() 
  train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc='Batches')
  for step, (inputs, context, targets) in enumerate(train_loader_tqdm):

    inputs = inputs.squeeze(dim=1).to(device)
    targets = targets.reshape(targets.shape[0],1).to(device)
    context = context.squeeze(dim=1).to(device)

    outputs = model(inputs, context)

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
    for step, (inputs, context, targets) in enumerate(test_loader_tqdm):

      inputs = inputs.squeeze(dim=1).to(device)
      targets = targets.reshape(targets.shape[0],1).to(device)
      context = context.squeeze(dim=1).to(device)

      outputs = model(inputs, context)

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

    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(-1, 1))
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
best_model_path = os.path.join(data_dir,'long_best.pth')
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
    train_loader = get_loader(train_df, id2emb, essay_embeddings, context_embeddings, shuffle=True)
    print("train_loader")
    test_loader = get_loader(test_df, id2emb, essay_embeddings, context_embeddings, shuffle=False)
    print("test_loader")
    # model
    print('------------------------------------------------------------------')
    print(f"\t\t\tTraining model: {n+1}")
    print('------------------------------------------------------------------')
    model = nn.DataParallel(MLP(input_size, embedding_size, window_size)).to(device)
    #model = Ngram_Clsfr().to(device)
    # loss and optimizer
    cost_function = nn.MSELoss()#CombinedLoss(alpha, beta, gamma)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # training
    train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
    test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
    print('Before training:\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))
    best_kappa = 0.0
    epoch_tqdm = tqdm(range(epochs), total=epochs, desc='Epochs')
    for epoch in epoch_tqdm:
        train_loss = training_step(model, cost_function, optimizer, train_loader)
        test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
        print("getting results df")
        results_df = get_results_df(train_df, test_df, test_preds)
        print("got results df")
        kappas_by_set = []

        kappas_by_set.append(
            kappa(
                results_df.loc[results_df["essay_set"] == essay_set, "score"],
                results_df.loc[results_df["essay_set"] == essay_set, "pred"],
                weights="quadratic",
            )
        )
        mean_kappa = np.mean(kappas_by_set)
        if mean_kappa > best_kappa:
            torch.save(model.state_dict(), best_model_path)
            print("Saving model")
            best_kappa = mean_kappa

        epoch_tqdm.set_postfix ({f"Mean Kappa: {mean_kappa:.5f} Best Kappa: {best_kappa:.5f} Test Loss: {test_loss:.5f} Train Loss: {train_loss:.5f} for Epoch: ":  {epoch+1}})

    model.load_state_dict(torch.load(best_model_path))
    train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
    test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
    print('After training:\t\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))

    print("getting results df")
    results_df = get_results_df(train_df, test_df, test_preds)
    print("got results df")
    kappas_by_set = []

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
    if n == 0:
        file = open(os.path.join(save_directory, f"qwk.txt"), "w")
    else:
        file = open(os.path.join(save_directory, f"qwk.txt"), "a")
    file.write(data)
    file.close()






