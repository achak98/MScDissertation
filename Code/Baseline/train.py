import torch
import torch.nn as nn
import torch.optim as optim
import evaluation
def train(model, train_dataloader, val_dataloader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average train loss
        train_loss /= len(train_dataloader)

        # Evaluate on validation set
        val_qwk = evaluation.evaluate(model, val_dataloader)

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val QWK: {val_qwk:.4f}")

    print("Training finished!")


