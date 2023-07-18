import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchcrf import CRF
from transformers import DebertaModel, T5ForConditionalGeneration, T5Tokenizer

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        query = self.query_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        attention_weights = F.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, -1)

        output = self.output_linear(attended_values)

        return output

class Biaffine(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Biaffine, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.randn(num_classes, input_dim, input_dim))
        self.b = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, 1, self.input_dim)
        Wx = torch.matmul(x, self.W)
        Wx = Wx.view(batch_size, seq_len, self.num_classes, self.input_dim)
        scores = torch.matmul(Wx, x.transpose(2, 3)).squeeze(3) + self.b
        return scores

class MLP(torch.nn.Module):
  
  def __init__(self, input_size):
    super(MLP, self).__init__()
    
    self.layers = torch.nn.Sequential(
      torch.nn.Linear(input_size, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(256, 96),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(96, 1)
    ) 

  def forward(self, x):
    return self.layers(x)

class GCNT5CRF(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(GCNT5CRF, self).__init__()

        self.encoder = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.attention = MultiheadAttention(hidden_dim, num_heads)
        self.decoder = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.biaffine = Biaffine(hidden_dim, hidden_dim//16)
        self.mlp = MLP(hidden_dim//16)

    def forward(self, x, edge_index, mask):
        # Encoder
        x_enc = self.encoder(x, mask).last_hidden_state

        # GCN
        x_gcn = self.gcn(x_enc, edge_index)
        x_gcn = F.relu(x_gcn)

        # Multi-Head Attention
        x_attention = self.attention(x_gcn.transpose(1, 2))

        # Decoder
        decoder_input_ids = torch.cat((torch.ones(x.size(0), 1).long().to(x.device), x_attention.argmax(dim=-1)), dim=1)
        decoder_outputs = self.decoder(decoder_input_ids=decoder_input_ids)

        # Biaffine Transformation
        scores = self.biaffine(decoder_outputs.logits)

        # MLP
        scores = self.mlp(scores)

        return scores
# Example usage
def train():
    # Hyperparameters
    hidden_dim = 768
    num_classes = 5
    num_layers = 2
    num_heads = 4
    dropout = 0.1

    # Dummy input data (replace with your own graph structure)
    batch_size = 16
    num_nodes = 10
    input_ids = torch.randint(0, 30522, (batch_size, num_nodes))
    attention_mask = torch.ones(batch_size, num_nodes, dtype=torch.long)
    edge_index = torch.randint(num_nodes, (2, num_nodes))  # Example random edge indices
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    # Create GCN-Longformer-CRF model
    model = GCNT5CRF(hidden_dim, num_classes, num_layers, num_heads, dropout)

    # Forward pass
    scores = model(input_ids, edge_index, mask)
    print(scores.shape)  # Shape: (batch_size, num_nodes, num_classes)

    # Training loop
    # ...

# Call the train function
train()
