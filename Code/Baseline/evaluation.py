from sklearn.metrics import cohen_kappa_score
import torch

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # Convert logits to predicted labels
            #_, predicted = torch.max(outputs, dim=0)

            # Append predictions and targets to lists
            print(f"preditions: {outputs}")
            print(f"labels: {labels}")
            #predictions.append(predicted.item())
            #targets.append(labels.item())
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(targets, predictions, weights='quadratic')

    return qwk
