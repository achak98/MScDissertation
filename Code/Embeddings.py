import torch
import models
import os
import numpy as np
from torch.autograd import Variable
import gensim.downloader as api
from transformers import BertModel

class Skipgram_Util:
    def __init__(self, args, vocab):
        self.vocab = vocab
        self.embedding_layer = models.SkipGram(len(vocab), args.embedding_dim)
        new_state_dict = {}
        new_state_dict["weight"] = torch.load(os.path.join(args.skipgram_file_path, \
                                 f"skipgram_weights{args.prompt}_{args.embedding_dim}.pth"))["embeddings.weight"]
        self.embedding_layer.embeddings.load_state_dict (new_state_dict)

    def get_vecs_by_tokens(self, padded_tokens, lower_case_backup=True):
        input_indices = [self.vocab[token] for token in padded_tokens]
        input_tensor = Variable(torch.LongTensor(input_indices))
        embeddings = self.embedding_layer.embeddings(input_tensor)
        return embeddings

class NAE:
    def __init__ (self, args, vocab):
        self.vocab = vocab

    def get_vecs_by_tokens(self, padded_tokens, lower_case_backup=True):
        indexed_data = []
        for token in padded_tokens:
            indexed_sentence = self.vocab[token]
            indexed_data.append(indexed_sentence)
        ret = torch.tensor(indexed_data)
        return ret
    
class Word2Vec:
    def __init__ (self):
        self.embedding = api.load("word2vec-google-news-300")
    def get_vecs_by_tokens(self, padded_tokens, lower_case_backup=True):
        indexed_data = np.array([[0] * 300])
        for token in padded_tokens:
            if token in self.embedding:
                indexed_sentence = [self.embedding[token]]
            else: 
                indexed_sentence = np.array([[0] * 300])
            indexed_data = np.append(indexed_data,indexed_sentence, axis = 0)
        indexed_data = indexed_data[1:]
        ret = torch.tensor(indexed_data)
        return ret
    
class BERT: 
    def __init__ (self):
        model_name = 'bert-base-uncased'  # Example model name
        self.bert_model = BertModel.from_pretrained(model_name)

    def get_vecs_by_tokens(self, padded_tokens, attn_mask, lower_case_backup=True): 
        input_ids = torch.tensor(padded_tokens).unsqueeze(0)  # Add batch dimension
        attn_mask = torch.tensor(attn_mask).unsqueeze(0)  # Add batch dimension
        outputs = self.bert_model(input_ids, attention_mask=attn_mask)
        embeddings = outputs.last_hidden_state
        print(embeddings)
        exit()
        return embeddings
