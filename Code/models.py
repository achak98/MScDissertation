import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, AutoConfig, RobertaForCausalLM


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embedded = self.embeddings(target)
        output = self.linear(embedded)
        return output

class Baseline(nn.Module):
    def __init__(self, args, strides, kernels):
        super(Baseline, self).__init__()
        #print("in: {} out: {} ks: {}".format(args.embedding_dim, args.cnnfilters, args.cnn_window_size_small))
        self.conv1 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_small, stride = strides[0])
        self.pool1 = nn.MaxPool1d(kernel_size=kernels[0], stride=strides[1])
        self.gru1 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_medium, stride=strides[2])
        self.pool2 = nn.MaxPool1d(kernel_size=kernels[1], stride=strides[3])
        self.gru2 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=args.dropout)

        self.conv3 = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.cnnfilters, kernel_size=args.cnn_window_size_large, stride=strides[4])
        self.pool3 = nn.MaxPool1d(kernel_size=kernels[2], stride=strides[5])
        self.gru3 = nn.GRU(args.cnnfilters, args.bgru_hidden_size, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(p=args.dropout)

        self.fc = nn.Linear(args.bgru_hidden_size*2*3,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=x.permute(0,2,1)
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        x1=x1.permute(0,2,1)
        _, h1 = self.gru1(x1) #x1 should be batch size, sequence length, input length
        h1 = torch.cat((h1[0, :, :], h1[1, :, :]), dim=1)
        h1 = self.dropout1(h1)

        x2 = self.conv2(x)
        x2 = self.pool2(x2)
        x2=x2.permute(0,2,1)
        _, h2 = self.gru2(x2)
        h2 = torch.cat((h2[0, :, :], h2[1, :, :]), dim=1)
        h2 = self.dropout1(h2)

        x3 = self.conv3(x)
        x3 = self.pool3(x3)
        x3=x3.permute(0,2,1)
        _, h3 = self.gru3(x3)
        h3 = torch.cat((h3[0, :, :], h3[1, :, :]), dim=1)
        h3 = self.dropout1(h3)

        h = torch.cat((h1, h2, h3), dim=1)
        h = self.fc(h)
        h = self.sigmoid(h)

        return h.squeeze()

class RobertaEncDec(nn.Module):
    def __init__(self):
        super(RobertaEncDec, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        decoder_config = AutoConfig.from_pretrained("roberta-base")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention=True
        self.decoder = RobertaModel.from_pretrained('roberta-base', config=decoder_config)
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1),  
            nn.Sigmoid()                      
        )
    
    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        weights = self.attention(encoder_output.last_hidden_state)
        context_vector = torch.sum(weights * encoder_output.last_hidden_state, dim=2)        
        decoder_output = self.decoder(input_ids=context_vector.long(), \
                attention_mask=attention_mask, \
                encoder_hidden_states=encoder_output.last_hidden_state).last_hidden_state
        
        weights = self.attention(decoder_output)
        context_vector = torch.sum(weights * decoder_output, dim=1)        
        h = self.regressor(context_vector)
        h = h.squeeze()
        if(torch.Tensor([1]).squeeze().size() == h.size()):
            h = h.unsqueeze(dim=0)
        return h
 
    
class BiRoberta(nn.Module):
    def __init__(self):
        super(BiRoberta, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        decoder_config = RobertaConfig.from_pretrained('roberta-base')
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder = RobertaModel.from_pretrained('roberta-base', config=decoder_config)
        self.output_layer = nn.Linear(self.decoder.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        decoder_output = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_output)[0]
        logits = self.output_layer(decoder_output)
        return logits
    
class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        self.output_layer = nn.Linear(self.encoder.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.output_layer(encoder_output)
        score = self.sigmoid(logits)
        return score
