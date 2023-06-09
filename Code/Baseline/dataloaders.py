import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file_path, prompt):
        df = pd.read_csv(file_path, sep='\t')  # Read the .tsv file
        self.data = df[df.iloc[:, 1] == prompt]
        self.prompt = prompt
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a sample from the dataset
        sample = self.data.iloc[idx]
        # Extract the input features and target label from the sample
        features = sample['text']
        target = sample['domain1_score']
        if(self.prompt == '2') :
            target += sample['domain2_score']
        # Convert the features and target to tensors
        features_tensor = torch.tensor(features)
        target_tensor = torch.tensor(target)

        return features_tensor, target_tensor

def create_data_loaders(dataDir, batch_size, prompt, shuffle=True, num_workers=0):
    data_file = dataDir+'train_set.tsv'

    dataset = MyDataset(data_file, prompt)
    length = len(dataset)
    train_dataset = dataset[:length*0.8]
    val_dataset = dataset[length*0.8:length*0.9]
    test_dataset = dataset[length*0.9:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
