import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torchcrf

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define the ADU recognition model
class ADURecognitionModel(nn.Module):
    def __init__(self, num_tags):
        super(ADURecognitionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(768, num_tags)
        self.crf = torchcrf.CRF(num_tags)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.hidden2tag(sequence_output)
        return logits
    
# Example usage
text = "This is an example sentence."

# Tokenize the input text
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
attention_mask = torch.ones_like(input_ids)  # All tokens are attended to

# Instantiate and load the ADU recognition model
model = ADURecognitionModel(num_tags=4)
model.load_state_dict(torch.load('adu_recognition_model.pth'))

# Make predictions
logits = model(input_ids, attention_mask)
predicted_labels = model.crf.decode(logits)

# Convert predicted labels back to original tokens
predicted_labels = predicted_labels[0]  # Remove batch dimension
predicted_tokens = [tokens[i] for i in range(len(tokens)) if predicted_labels[i] != 0]

print(predicted_tokens)
