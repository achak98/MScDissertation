import torch
import numpy as np
import torch.nn as nn

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    targets = []
    loss_fn = nn.MSELoss()
    eval_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.float())
            eval_loss += loss.item()
            # Convert logits to predicted labels
            #_, predicted = torch.max(outputs, dim=0)

            # Append predictions and targets to lists
            #predictions.append(predicted.item())
            #targets.append(labels.item())
            predictions.extend((outputs*10 + 2).cpu().numpy())
            targets.extend((labels*10 + 2).cpu().numpy())

            #print(f"preditions: {predictions}")
            #print(f"targets: {targets}")
    eval_loss /= len(dataloader)

    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(targets, predictions)

    return qwk, eval_loss

class CustomLoss(nn.Module):
    def __init__(self, prompt):
        super(CustomLoss, self).__init__()
        self.prompt = prompt 
    def forward(self, output, target):
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        predictions = (denormalise_scores(self.prompt, output))
        targets = (denormalise_scores(self.prompt, target))
        qwk = quadratic_weighted_kappa(targets, predictions)
        return loss + (1-qwk)*10
    
def evaluate_roberta(model, dataloader, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    targets = []
    loss_fn = CustomLoss(prompt)
    eval_loss = 0.0
    with torch.no_grad():
        for batch_num, batch in enumerate(dataloader):
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
            scores = batch['score'].float().to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, scores)
            eval_loss += loss.item()

            predictions.extend(denormalise_scores(prompt, outputs))
            targets.extend(denormalise_scores(prompt, scores))
            
    eval_loss /= len(dataloader)

    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(targets, predictions)
    print(f"eval loss: {eval_loss:5f} qwk scores: {qwk}")
    return qwk, eval_loss

def denormalise_scores(prompt, undetached_data):
    data = undetached_data.detach().clone()

    if prompt == 1:
        data = (data * 10 + 2).cpu().numpy()
    elif prompt == 2:
        data = (data * 8 + 2).cpu().numpy()
    elif prompt == 3:
        data = (data * 3).cpu().numpy()
    elif prompt == 4:
        data = (data * 3).cpu().numpy()
    elif prompt == 5:
        data = (data * 4).cpu().numpy()
    elif prompt == 6:
        data = (data * 4).cpu().numpy()
    elif prompt == 7:
        data = (data * 30).cpu().numpy()
    elif prompt == 8:
        data = (data * 60).cpu().numpy()
    return data

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Convert predictions to integer values
    y_pred = np.array((y_pred).round(decimals=0, out=None))

    # Convert targets to integer values if needed
    y_true = np.array((y_true).round(decimals=0, out=None))
    #print(f"y_pred: {y_pred}")
    #print(f"y_true: {y_true}")
    # Calculate the confusion matrix
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    conf_mat = calculate_confusion_matrix(min_rating, max_rating, y_true, y_pred)

    # Create the weight matrix
    num_ratings = len(conf_mat)
    num_scored_items = float(len(y_true))
    """weight_matrix = np.zeros((num_ratings, num_ratings))
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
    print(f"denominator: {denominator}")
    print(f"numerator: {numerator}")"""

    hist_rater_a = histogram(y_true, min_rating, max_rating)
    hist_rater_b = histogram(y_pred, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    if num_ratings == 1:
        if y_pred[0] == y_true[0]:
            return 1
        else:
            return 0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    # Calculate QWK
    qwk = 1.0 - (numerator / denominator)

    return qwk

def calculate_confusion_matrix(min_rating, max_rating, y_true, y_pred):
    
    num_classes = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_classes)]
                for j in range(num_classes)]

    for true, pred in zip(y_true, y_pred):
        conf_mat[int(true - min_rating)][ int(pred - min_rating)] += 1

    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[int(r - min_rating)] += 1
    return hist_ratings