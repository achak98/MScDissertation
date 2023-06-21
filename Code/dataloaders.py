import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.tokenize import word_tokenize, sent_tokenize
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, GloVe
from transformers import BertTokenizer
import Embeddings

class MyDataset(Dataset):
    def __init__(self, file_path, prompt, max_length, embedding, tokeniser):
        df = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1')
        model_name = 'bert-base-uncased'
        self.data = df[df.iloc[:, 1] == prompt]
        self.max_length = max_length
        self.prompt = prompt
        self.embedding = embedding
        self.tokeniser = tokeniser
        self.bert_tokeniser = BertTokenizer.from_pretrained(model_name)
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
        padded_tokens, attn_mask = self.tokenize(essay)
        if(self.tokeniser == "nltk"):
            ret = self.embedding.get_vecs_by_tokens(padded_tokens, lower_case_backup=True)
        elif(self.tokeniser == "bert"):
            ret = self.embedding.get_vecs_by_tokens(padded_tokens, attn_mask, lower_case_backup=True)
        target_tensor = torch.tensor(score)  # Convert target to tensor
        return ret, target_tensor
    
    def tokenize(self, essay):
        if self.tokeniser == "nltk":
            tokens = word_tokenize(essay)
            padded_tokens = self.pad_sequence(tokens)
            attn_mask = None
            return padded_tokens, attn_mask
        elif self.tokeniser == "bert":
            sents = sent_tokenize(essay)
            tokens = []
            #NEED TO ADD TOKENS AFTER TOKENISED
            sents[0] = [self.bert_tokeniser.cls_token] + sents[0] + [self.bert_tokeniser.sep_token]
            for i in range(1, len(sents)):
                sents[i] = sents[i] + [self.bert_tokeniser.sep_token]
            sents[-1] = sents[-1] + [self.bert_tokeniser.cls_token]

            for sent in sents:
                tokens.extend(self.bert_tokeniser.tokenize(sent))
            #ADD THE TOKENS HERE EG: 
            """text = "This is an example sentence."
            tokens = tokenizer.tokenize(text)
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)"""
            token_ids = self.bert_tokeniser.convert_tokens_to_ids(tokens)
            print(f"token_ids: {token_ids}")
            padding_id = self.bert_tokeniser.pad_token_id
            attn_mask = [1] * len(token_ids)  # Attention mask to differentiate padded tokens

            if len(token_ids) < self.max_length:
                padding_length = self.max_length - len(token_ids)
                token_ids = token_ids + [padding_id] * padding_length
                attn_mask = attn_mask + [0] * padding_length
            else:
                token_ids = token_ids[:self.max_length]
                attn_mask = attn_mask[:self.max_length]
            print(f"padded token_ids: {token_ids}")
            return token_ids, attn_mask

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_length:
            sequence = sequence + ['na'] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        return sequence


def get_vocab_and_dataset_length(data_file, prompt):
    df = pd.read_csv(data_file, sep='\t', encoding='ISO-8859-1')
    data = df[df.iloc[:, 1] == prompt]
    concatenated_text = data['essay'].str.cat(sep=' ')
    tokens = word_tokenize(concatenated_text)
    tokens.append("na")
    vocab = build_vocab_from_iterator([tokens])
    return vocab, len(data)

def create_data_loaders(args, embedding_type, shuffle=True, num_workers=0):
    data_file = args.dataDir + '/training_set_rel3.tsv'
    
    vocab, dataset_length = get_vocab_and_dataset_length(data_file, args.prompt)
    
    split_lengths = [int(dataset_length * 0.7), int(dataset_length * 0.1), \
                     dataset_length - int(dataset_length * 0.7) - int(dataset_length * 0.1)]

    if (embedding_type == "glove"):
        embedding = GloVe(name='6B', dim=args.embedding_dim)
        tokeniser = "nltk"
    elif (embedding_type == "w2v"):
        embedding = Embeddings.Word2Vec()
        tokeniser = "nltk"
    elif (embedding_type == "NAE"):
        embedding = Embeddings.NAE(args, vocab)
        tokeniser = "nltk"
        split_lengths = [dataset_length, 0, 0]
    elif (embedding_type == "skipgram"):
        embedding = Embeddings.Skipgram_Util(args, vocab)
        tokeniser = "nltk"
    elif (embedding_type == "bert"):
        embedding = Embeddings.BERT()
        tokeniser = "bert"
    elif (embedding_type == "droberta"):
        embedding = "hehelolzzzzz(2)"
        tokeniser = "roberta"
    
    
    dataset = MyDataset(data_file, args.prompt, args.max_length, embedding, tokeniser)
    slices = random_split(dataset, split_lengths)

    train_dataset = slices[0]
    val_dataset = slices[1]
    test_dataset = slices[2]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(vocab)
