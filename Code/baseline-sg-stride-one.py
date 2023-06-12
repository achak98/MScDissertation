import argsparser, dataloaders, evaluation, train, models

import random
import torch
import numpy as np

def main():

    random.seed(42)

    qwk_score = []
    seeds = []

    for i in range (10):
        seed = random.randint(1, 1000)

        # Set the random seed for Python's random module
        random.seed(seed)

        # Set the random seed for NumPy
        np.random.seed(seed)

        # Set the random seed for PyTorch
        torch.manual_seed(seed)

        # Set the random seed for CUDA operations (if using GPUs)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set the deterministic behavior for cudNN (if using GPUs)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Parse the arguments
        args = argsparser.parse_args()

        qwk_score.append(baseline(args))
    file_path = "/home/achakravarty/Dissertation/Data/bs-sg-one/qwk.txt"
    file = open(file_path, "w")

    # Write data to the file
    data = ""

    # Loop through the zipped pairs
    for seed, score in zip(seeds, qwk_score):
        data += f"For seed {seed}, score was {score} \n" 
    file.write(data)

    # Close the file
    file.close()

def baseline(args):
    print("!!========================CREATING DATA LOADERS========================!!")
    train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
    .create_data_loaders(args, embedding_type = "skipgram")
    
    #print("VOCAB_SIZE: ", vocab_size)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline_Stride_One(args)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score_for_seed =  evaluation.evaluate(model, test_dataloader)
    print("Quadratic Weighted Kappa (QWK) Score on test set:", qwk_score_for_seed)
    return qwk_score_for_seed

if __name__ == "__main__":
    main()