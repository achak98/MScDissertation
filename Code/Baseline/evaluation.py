import torch
import numpy as np

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
            #predictions.append(predicted.item())
            #targets.append(labels.item())
            predictions.extend((outputs*12).cpu().numpy())
            targets.extend((labels*12).cpu().numpy())

            print(f"preditions: {predictions}")
            print(f"targets: {targets}")
    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(targets, predictions)

    return qwk


def quadratic_weighted_kappa(y_true, y_pred):

    # Convert predictions to integer values
    y_pred = np.round(y_pred).astype(int)

    # Convert targets to integer values if needed
    y_true = np.round(y_true).astype(int)

    # Calculate the confusion matrix
    conf_mat = calculate_confusion_matrix(y_true, y_pred)

    # Create the weight matrix
    num_ratings = conf_mat.shape[0]
    weight_matrix = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight_matrix[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    # Calculate observed and expected matrices
    obs_mat = calculate_confusion_matrix(y_true, y_pred)
    exp_mat = np.outer(np.sum(obs_mat, axis=1), np.sum(obs_mat, axis=0))

    # Normalize observed and expected matrices
    obs_mat = obs_mat / np.sum(obs_mat)
    exp_mat = exp_mat / np.sum(exp_mat)

    # Calculate the numerator and denominator of QWK
    numerator = np.sum(weight_matrix * obs_mat)
    denominator = np.sum(weight_matrix * exp_mat)

    # Calculate QWK
    qwk = 1.0 - (numerator / denominator)

    return qwk

def calculate_confusion_matrix(y_true, y_pred):
    num_classes = max(max(y_true), max(y_pred)) + 1
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        conf_mat[true, pred] += 1

    return conf_mat