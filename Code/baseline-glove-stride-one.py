import argsparser, dataloaders, evaluation, train, models

import random
import torch
import numpy as np

# Set the random seed for Python's random module
random.seed(42)

# Set the random seed for NumPy
np.random.seed(42)

# Set the random seed for PyTorch
torch.manual_seed(42)

# Set the random seed for CUDA operations (if using GPUs)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set the deterministic behavior for cudNN (if using GPUs)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parse the arguments
args = argsparser.parse_args()

def baseline():
    print("!!========================CREATING DATA LOADERS========================!!")
    train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
    .create_data_loaders(args, embedding_type = "glove")
    
    #print("VOCAB_SIZE: ", vocab_size)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline_Stride_One(args)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score = evaluation.evaluate(model, test_dataloader)
    print("Quadratic Weighted Kappa (QWK) Score on test set:", qwk_score)

if __name__ == "__main__":
    baseline()
