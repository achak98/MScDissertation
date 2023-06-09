from sklearn.metrics import cohen_kappa_score

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
            _, predicted = torch.max(outputs, dim=1)

            # Append predictions and targets to lists
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(targets, predictions, weights='quadratic')

    return qwk