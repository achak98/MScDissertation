import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, GloVe
import Embeddings

class MyDataset(Dataset):
    def __init__(self, file_path, prompt, max_length, embedding):
        df = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1')
        self.data = df[df.iloc[:, 1] == prompt]
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = max_length
        self.prompt = prompt
        self.embedding = embedding
        #self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        essay = sample['essay']
        score = sample['domain1_score']
        if self.prompt == 1:
            score = (score-2)/10
        elif self.prompt == 2:
            score += sample['domain2_score']
        tokens = self.tokenizer(essay)
        padded_tokens = self.pad_sequence(tokens)
        ret = self.embedding.get_vecs_by_tokens(padded_tokens, lower_case_backup=True)
        target_tensor = torch.tensor(score)  # Convert target to tensor
        return ret, target_tensor

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_length:
            sequence = sequence + ['<UNK>'] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        return sequence

def collate_fn(batch):
    features, targets = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True)
    padded_targets = torch.stack(targets)  # Stack the target tensors
    return padded_features, padded_targets

def get_vocab_and_dataset_length(data_file, prompt):
    tokeniser = get_tokenizer('basic_english')
    df = pd.read_csv(data_file, sep='\t', encoding='ISO-8859-1')
    data = df[df.iloc[:, 1] == prompt]
    concatenated_text = data['essay'].str.cat(sep=' ')
    tokens = tokeniser(concatenated_text)
    tokens.append("<UNK>")
    vocab = build_vocab_from_iterator([tokens])
    return vocab, len(data)

def create_data_loaders(args, embedding_type, shuffle=True, num_workers=0):
    data_file = args.dataDir + '/training_set_rel3.tsv'
    
    vocab, dataset_length = get_vocab_and_dataset_length(data_file, args.prompt)
    
    split_lengths = [int(dataset_length * 0.7), int(dataset_length * 0.1), \
                     dataset_length - int(dataset_length * 0.7) - int(dataset_length * 0.1)]

    if (embedding_type == "glove"):
        embedding = GloVe(name='6B', dim=args.embedding_dim)
    elif (embedding_type == "w2v"):
        embedding = Embeddings.Word2Vec()
    elif (embedding_type == "NAE"):
        embedding = Embeddings.NAE(args, vocab)
        split_lengths = [dataset_length, 0, 0]
    elif (embedding_type == "skipgram"):
        embedding = Embeddings.Skipgram_Util(args, vocab)
    elif (embedding_type == "droberta"):
        embedding = "hehelolzzzzz(2)"
    
    
    dataset = MyDataset(data_file, args.prompt, args.max_length, embedding)
    slices = random_split(dataset, split_lengths)

    train_dataset = slices[0]
    val_dataset = slices[1]
    test_dataset = slices[2]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    #                          collate_fn=collate_fn)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                        collate_fn=collate_fn)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                         collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, len(vocab)
