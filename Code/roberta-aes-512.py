import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score as kappa
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original kaggle training set
kaggle_dataset = pd.read_csv(
    "./../Data//ASAP-AES/training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1"
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


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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

roberta = RobertaModel.from_pretrained("roberta-base").to(device)



def mean_encoding(essay_list, model, tokenizer):

  print('Encoding essay embeddings:')

  embeddings = []
  for essay in tqdm(essay_list):
    encoded_input = tokenizer(essay, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
      model_output = model(**encoded_input)
    tokens_embeddings = np.matrix(model_output[0].squeeze().cpu())
    embeddings.append(np.squeeze(np.asarray(tokens_embeddings)))

  return np.matrix(embeddings)


essay_embeddings = mean_encoding(dataset['essay'], roberta, tokenizer)



def get_loader(df, id2emb, essay_embeddings, shuffle=True):

  # get embeddings from essay_id using id2emb dict
  embeddings = np.array([essay_embeddings[id2emb[id]] for id in df['essay_id']])

  # dataset and dataloader
  data = TensorDataset(torch.from_numpy(embeddings).float(), torch.from_numpy(np.array(df['scaled_score'])).float())
  loader = DataLoader(data, batch_size=128, shuffle=shuffle, num_workers=2)

  return loader



class MLP(torch.nn.Module):
  
  def __init__(self, input_size):
    super(MLP, self).__init__()
    
    self.layers1 = torch.nn.Sequential(
      torch.nn.Linear(input_size, 768),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(96, 1)
    )
    self.lstm1 = nn.LSTM(input_size, input_size//2, num_layers=1, bidirectional=True)
    self.dropout1 = nn.Dropout(0.3) 
    self.fc1 = nn.Linear(input_size,input_size//2)
    self.dropout2 = nn.Dropout(0.3)
    self.attention_weights = nn.Linear(input_size//2 * 3, 1)
    self.dropout3 = nn.Dropout(0.3) 
    self.lstm2 = nn.LSTM(input_size, input_size//2, num_layers=1, bidirectional=True)
    self.dropout4 = nn.Dropout(0.3)
    self.fc1 = nn.Linear(input_size,input_size//2) 
    self.dropout5 = nn.Dropout(0.3)
    self.layers2 = torch.nn.Sequential(
      torch.nn.Linear(input_size//2, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(96, 1)
    ) 
    def similarity(self, hi, hj):
        # Concatenate the hidden representations
        h_concat = torch.cat([hi, hj, hi * hj], dim=-1)
        return self.attention_weights(h_concat)
    def forward(self, x):
        layer_1_out = self.layers1(x)
        print("layer_1_out: ",layer_1_out.size())
        layer_1_out = layer_1_out.squeeze()
        print("layer_1_out squeezed: ",layer_1_out.size())
        lstm_out, _ = self.lstm1(layer_1_out)
        print("lstm_out: ",lstm_out.size())
        lstm_out = self.dropout1(lstm_out)
        lstm_out_sum = self.fc1(lstm_out)
        lstm_out_sum = self.dropout2 (lstm_out_sum)
        print("lstm_out_sum: ",lstm_out_sum.size())
        batch_size, seq_length, hidden_dim = lstm_out_sum.size()
        # Initialize attention vector tensor
        attention_vectors = torch.zeros_like(lstm_out_sum)
        for i in range(seq_length):
            # Define the start and end positions of the window
            start_pos = max(0, i - self.window_size)
            end_pos = min(seq_length, i + self.window_size + 1)
            
            # Compute similarity between the current word and nearby words
            similarity_scores = torch.cat([self.similarity(lstm_out_sum[:, i], lstm_out_sum[:, j]) for j in range(start_pos, end_pos)], dim=1)

            attention_weights = torch.nn.functional.softmax(similarity_scores, dim=-1) #this has all alpha(i,j)s

            attention_vector = torch.sum((lstm_out_sum[:, start_pos:end_pos, :].permute(2,0,1) * attention_weights).permute(1,2,0), dim=1)

            attention_vectors[:,i] = attention_vector.squeeze(1) #(seqlen,hiddim)
        print("attention_vectors: ",attention_vectors.size())
        lstm_output_with_attention = torch.cat([lstm_out_sum, attention_vectors], dim=-1)
        lstm_output_with_attention = self.dropout3(lstm_output_with_attention)
        print("lstm_output_with_attention: ",lstm_output_with_attention.size())
        lstm_out2 = self.lstm2(lstm_output_with_attention)
        lstm_out2 = self.dropout4(lstm_out2)
        print("lstm_out2: ",lstm_out2.size())
        lstm_out_sum2 = self.fc1(lstm_out2)
        lstm_out_sum2 = self.dropout5(lstm_out_sum2)
        print("lstm_out_sum2: ",lstm_out_sum2.size())
        layer_2_out = self.layers2(lstm_out_sum2)
        print("layer_2_out: ",layer_2_out.size())
        return layer_2_out


def training_step(model, cost_function, optimizer, train_loader):

  samples = 0.
  cumulative_loss = 0.

  model.train() 

  for step, (inputs, targets) in enumerate(train_loader):

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
    for step, (inputs, targets) in enumerate(test_loader):

      inputs = inputs.squeeze(dim=1).to(device)
      targets = targets.reshape(targets.shape[0],1).to(device)

      outputs = model(inputs)

      loss = cost_function(outputs, targets)

      samples += inputs.shape[0]
      cumulative_loss += loss.item()
      for out in outputs:
        preds.append(float(out))

  return cumulative_loss/samples, preds
     
# hyper-parameters
input_size = 768
epochs = 20
lr = 3e-4

# cross-validation folds
kf = KFold(n_splits=10, random_state=2022, shuffle=True)

# dicts with train_df, test_df and predictions for each model
train_df_dict = {}
test_df_dict = {}
preds_dict = {}

# copy of dataset with scaled scores computed using the whole dataset
scaled_dataset = get_scaled_dataset(dataset)

for n, (train, test) in enumerate(kf.split(dataset)):
  
  # train, test splits 
  # scaled scores in train_df are computed only using training data
  train_df = dataset.iloc[train]
  train_df = get_scaled_dataset(train_df)

  test_df = scaled_dataset.iloc[test]

  # dataloaders
  train_loader = get_loader(train_df, id2emb, essay_embeddings, shuffle=True)
  test_loader = get_loader(test_df, id2emb, essay_embeddings, shuffle=False)

  # model
  print('------------------------------------------------------------------')
  print(f"\t\t\tTraining model: {n+1}")
  print('------------------------------------------------------------------')
  model = MLP(input_size).to(device)
  
  # loss and optimizer
  cost_function = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # training
  train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
  test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
  print('Before training:\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))

  for epoch in range(epochs):
    train_loss = training_step(model, cost_function, optimizer, train_loader)
    test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
    print('Epoch: {:}\t\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(epoch+1,train_loss, test_loss))

  train_loss, train_preds = test_step(model, cost_function, optimizer, train_loader)
  test_loss, test_preds = test_step(model, cost_function, optimizer, test_loader)
  print('After training:\t\tLoss/train: {:.5f}\tLoss/test: {:.5f}'.format(train_loss, test_loss))

  # store test_df and predictions
  train_df_dict[f"model_{n+1}"] = train_df
  test_df_dict[f"model_{n+1}"] = test_df
  preds_dict[f"model_{n+1}"] = test_preds
  


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


# list of mqw_kappa for each model
mqwk_list = []

for model_id in test_df_dict:
    results_df = get_results_df(
        train_df_dict[model_id], test_df_dict[model_id], preds_dict[model_id]
    )

    kappas_by_set = []
    for essay_set in range(1, 9):
        kappas_by_set.append(
            kappa(
                results_df.loc[results_df["essay_set"] == essay_set, "score"],
                results_df.loc[results_df["essay_set"] == essay_set, "pred"],
                weights="quadratic",
            )
        )

    mqwk_list.append(np.mean(kappas_by_set))
print("----------------------------------------------")
print("mean QWK after 5-fold cross-validation:\t{:.4f}".format(np.mean(mqwk_list)))
print("Max mean QWK:\t\t\t\t{:.4f}".format(max(mqwk_list)))
print("Min mean QWK:\t\t\t\t{:.4f}".format(min(mqwk_list)))

raters_kappas = []
for essay_set in range(1, 9):
    raters_kappas.append(
        kappa(
            dataset.loc[dataset["essay_set"] == essay_set, "rater1"],
            dataset.loc[dataset["essay_set"] == essay_set, "rater2"],
            weights="quadratic",
        )
    )

mqwk_raters = np.mean(raters_kappas)
print("----------------------------------------------")
print("mean QWK for two human raters:\t\t{:.4f}".format(mqwk_raters))
print("----------------------------------------------")


def show_results(id, data):
    model_id = f"model_{id}"
    results_df = get_results_df(
        train_df_dict[model_id], test_df_dict[model_id], preds_dict[model_id]
    )

    kappas_by_set = []
    for essay_set in range(1, 9):
        kappas_by_set.append(
            kappa(
                results_df.loc[results_df["essay_set"] == essay_set, "score"],
                results_df.loc[results_df["essay_set"] == essay_set, "pred"],
                weights="quadratic",
            )
        )
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

    return results_df, data


def plot_results(results_df, id):
    set_number = 0
    fig, ax = plt.subplots(4, 2, figsize=(9, 9), sharey=False)
    for i in range(4):
        for j in range(2):
            set_number += 1
            results_df[results_df["essay_set"] == set_number][
                ["score", "pred"]
            ].plot.hist(histtype="step", bins=20, ax=ax[i, j], rot=0)
            ax[i, j].set_title("Set %i" % set_number)
    ax[3, 0].locator_params(nbins=10)
    ax[3, 1].locator_params(nbins=10)
    plt.suptitle(f"Histograms of scores for Model {id}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


data = ""
for i in range(1, 11):
    results_df, data = show_results(i, data)

file = open("./../Data/results/roberta-finetune/qwk.txt", "w")
file.write(data)
file.close()
