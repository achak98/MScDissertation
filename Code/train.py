import torch
import torch.nn as nn
import torch.optim as optim
import evaluation
from torch.autograd import Variable
from tqdm import tqdm
import os

def train(model, train_dataloader, val_dataloader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Train the model with tqdm progress bars
    with tqdm(total=num_epochs, desc='Epochs', unit='epoch') as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_pbar.set_description(f'Epoch {epoch+1}')
            epoch_pbar.update()

            model.train()
            train_loss = 0.0

            with tqdm(total=len(train_dataloader), desc='Batches', unit='batch') as batch_pbar:
                for batch_num, (inputs, labels) in enumerate(train_dataloader):
                    batch_pbar.set_description(f'Batch {batch_num+1}')
                    batch_pbar.update()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs.float())

                    # Calculate loss
                    loss = loss_fn(outputs, labels.float())

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    batch_pbar.set_postfix({'Loss': loss.item()})

            # Calculate average train loss
            train_loss /= len(train_dataloader)

            # Evaluate on validation set
            val_qwk, val_loss = evaluation.evaluate(model, val_dataloader)

            # Print epoch statistics
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Val QWK: {val_qwk:.4f}")


        print("Training finished!")
        return model



def train_skipGram (dataloader, model, vocab_size, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    with tqdm(total=args.num_epochs, desc='Epochs', unit='epoch') as epoch_pbar:
        for epoch in range(args.num_epochs):
            epoch_pbar.set_description(f'Epoch {epoch+1}')
            epoch_pbar.update()
            with tqdm(total=len(dataloader), desc='Batches', unit='batch') as batch_pbar:
                for batch_num, data in enumerate(dataloader):
                    batch_pbar.set_description(f'Batch {batch_num+1}')
                    batch_pbar.update()
                    data = data[0]
                    #print("here 2")
                    target = data.to(device)
                    #print("here 3")
                    output = model(target)
                    #print("here 4")
                    # Reshape the output and target tensors for the loss calculation
                    output = output.view(-1, vocab_size)
                    #print("here 5")
                    target = target.view(-1)
                    #print("here 6")
                    # Compute the loss and perform backpropagation
                    loss = criterion(output, target)
                    #print("here 7")
                    optimizer.zero_grad()
                    #print("here 8")
                    loss.backward()
                    #print("here 9")
                    optimizer.step()
                    #print("here 10")
                    torch.save(model.state_dict(), \
            os.path.join(args.skipgram_file_path, \
                            f"skipgram_weights{args.prompt}_{args.embedding_dim}_{epoch}.pth"))
    
    return model

def train(model, train_dataloader, val_dataloader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Train the model with tqdm progress bars
    with tqdm(total=num_epochs, desc='Epochs', unit='epoch') as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_pbar.set_description(f'Epoch {epoch+1}')
            epoch_pbar.update()

            model.train()
            train_loss = 0.0

            with tqdm(total=len(train_dataloader), desc='Batches', unit='batch') as batch_pbar:
                for batch_num, (inputs, labels) in enumerate(train_dataloader):
                    batch_pbar.set_description(f'Batch {batch_num+1}')
                    batch_pbar.update()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs.float())

                    # Calculate loss
                    loss = loss_fn(outputs, labels.float())

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    batch_pbar.set_postfix({'Loss': loss.item()})

            # Calculate average train loss
            train_loss /= len(train_dataloader)

            # Evaluate on validation set
            val_qwk, val_loss = evaluation.evaluate(model, val_dataloader)

            # Print epoch statistics
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Val QWK: {val_qwk:.4f}")


        print("Training finished!")
        return model



def train_roberta(model, train_dataloader, val_dataloader, num_epochs, lr, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Train the model with tqdm progress bars
    with tqdm(total=num_epochs, desc='Epochs', unit='epoch') as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_pbar.set_description(f'Epoch {epoch+1}')
            epoch_pbar.update()

            model.train()
            train_loss = 0.0
            
            with tqdm(total=len(train_dataloader), desc='Batches', unit='batch') as batch_pbar:
                for batch_num, (batch) in enumerate(train_dataloader):
                    batch_pbar.set_description(f'Batch {batch_num+1}')
                    batch_pbar.update()
                    input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
                    attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
                    scores = batch['score'].float().to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    if(batch_num > len(train_dataloader)*0.99):
                        print(f"scores: {scores}")
                        print(f"outputs: {outputs}")
                    # Calculate loss
                    loss = loss_fn(outputs, scores)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    batch_pbar.set_postfix({})
                    batch_pbar.set_postfix({f'Last Loss: {loss.item()} and Running Avg Loss:' : (train_loss/(batch_num+1))})
            # Calculate average train loss
            train_loss /= len(train_dataloader)
            
            # Evaluate on validation set
            val_qwk, val_loss = evaluation.evaluate_roberta(model, val_dataloader, prompt)

            # Print epoch statistics
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Val QWK: {val_qwk:.4f}")


        print("Training finished!")
        return model