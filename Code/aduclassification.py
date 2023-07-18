import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Custom dataset class
class ADUDataset(Dataset):
    def __init__(self, text_dir, adu_dir, tokenizer, max_length):
        self.text_dir = text_dir
        self.adu_dir = adu_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts, self.labels = self.load_data()

    def load_data(self):
        texts = []
        labels = []

        for filename in os.listdir(self.text_dir):
            if filename.endswith('.txt'):
                text_path = os.path.join(self.text_dir, filename)
                adu_path = os.path.join(self.adu_dir, filename.replace('.txt', '_adu.txt'))

                with open(text_path, 'r') as text_file, open(adu_path, 'r') as adu_file:
                    text = text_file.read().strip()
                    adus = adu_file.readlines()

                    # Process ADUs and labels
                    for adu in adus:
                        adu = adu.strip()
                        if adu:
                            adu_text, adu_label = adu.split('\t')
                            texts.append(adu_text)
                            labels.append(adu_label)

        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(int(label))
        }

# Example usage
def train():
    text_dir = 'path/to/texts'
    adu_dir = 'path/to/adus'

    # Hyperparameters
    batch_size = 16
    max_length = 128
    num_epochs = 5
    learning_rate = 2e-5

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create dataset and data loader
    dataset = ADUDataset(text_dir, adu_dir, tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed.")

    # Save the trained model
    model.save_pretrained("trained_model")

# Call the train function
train()
