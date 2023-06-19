import argsparser, dataloaders, evaluation, train, models

import random
import torch
import numpy as np

def main():

    embedding_types = ["glove", "skipgram", "w2v"]
    file_path = "/home/achakravarty/Dissertation/Data/results/qwk.txt"

    for embedding_type in embedding_types:
        for stride1 in range(1, args.cnn_window_size_small+1):
            for stride2 in range (1, args.cnn_window_size_small+1):
                for stride3 in range (1, args.cnn_window_size_medium+1):
                    for stride4 in range (1, args.cnn_window_size_medium+1):
                        for stride5 in range (1, args.cnn_window_size_large+1):
                            for stride6 in range (1, args.cnn_window_size_large+1):
                                qwk_score = []
                                seeds = []
                                random.seed(42)
                                # Write data to the file
                                data = f"For embedding type: {embedding_type} stride1: {stride1} stride2:{stride2} stride3: {stride3} stride4: {stride4} stride5: {stride5} stride6: {stride6} \n"

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
                                    qwk_score.append(69420)
                                    #qwk_score.append(baseline(args, embedding_type, stride1, stride2, stride3, stride4, stride5, stride6))
                                    seeds.append(seed)

                                file = open(file_path, "a")

                                # Loop through the zipped pairs
                                for seed, score in zip(seeds, qwk_score):
                                    data += f"For seed {seed}, score was {score} \n" 
                                data += f"Average QWK score: {np.average(qwk_score)} \n"
                                file.write(data)

                                # Close the file
                                file.close()
    

def baseline(args, embedding_type, stride1, stride2, stride3, stride4, stride5, stride6):
    print("!!========================CREATING DATA LOADERS========================!!")
    train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
    .create_data_loaders(args, embedding_type = embedding_type)
    
    #print("VOCAB_SIZE: ", vocab_size)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline(args, stride1, stride2, stride3, stride4, stride5, stride6)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score_for_seed =  evaluation.evaluate(model, test_dataloader)
    print("Quadratic Weighted Kappa (QWK) Score on test set:", qwk_score_for_seed)
    return qwk_score_for_seed

if __name__ == "__main__":
    main()
    
