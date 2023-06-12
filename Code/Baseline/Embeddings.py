import torch
import models
import os
from torch.autograd import Variable

class Skipgram_Util:
    def __init__(self, args, vocab):
        self.vocab = vocab
        self.embedding_layer = models.SkipGram(len(vocab), args.embedding_dim)
        self.embedding_layer.embeddings.load_state_dict \
        (torch.load(os.path.join(args.skipgram_file_path, 
                                 f"skipgram_weights{args.prompt}_{args.embedding_dim}.pth")))

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