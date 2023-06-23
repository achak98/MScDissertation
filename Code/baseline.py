import argsparser, dataloaders, evaluation, train, models

import random
import torch
import numpy as np

def main():

    embedding_types = ["w2v"] #["glove", "skipgram", "w2v"]
    file_path = "/home/achakravarty/Dissertation/Data/results/qwk.txt"
    args = argsparser.parse_args()
    for embedding_type in embedding_types:
        """
        for stride1 in range(args.cnn_window_size_small, 1, -1):
            for ks1 in range(args.cnn_window_size_small, 1, -1):
                for stride2 in range (args.cnn_window_size_small, 1, -1):
                    for stride3 in range (args.cnn_window_size_medium, 1, -1):
                        for ks2 in range (args.cnn_window_size_medium, ks1, -1):
                            for stride4 in range (args.cnn_window_size_medium, ks1, -1):
                                for stride5 in range (args.cnn_window_size_large, 1, -1):
                                    for ks3 in range (args.cnn_window_size_large, ks2, -1):
                                        for stride6 in range (args.cnn_window_size_large, ks2, -1):
                                        """
        qwk_score = []
        test_loss = []
        seeds = []
        random.seed(42)
        # Write data to the file
        #data = f"For embedding type: {embedding_type} stride1: {stride1} ks1: {ks1} stride2:{stride2} stride3: {stride3} ks2: {ks2} stride4: {stride4} stride5: {stride5} ks3: {ks3} stride6: {stride6} \n"
        stride1 = 2
        ks1 = 2
        stride2 = 2
        ks2 = 3
        stride3 = 3
        stride4 =3
        ks3 = 4
        stride5 = 4
        stride6= 4
        
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

            print("!!========================CREATING DATA LOADERS========================!!")
            train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
                .create_data_loaders(args, embedding_type = embedding_type)
            dataloaders_list = [train_dataloader, val_dataloader, test_dataloader]
            # Parse the arguments
            qwk_score_for_seed, test_loss_for_seed = baseline(args, dataloaders_list, embedding_type, stride1, stride2, stride3, stride4, stride5, stride6, ks1, ks2, ks3)
            qwk_score.append(qwk_score_for_seed)
            test_loss.append(test_loss_for_seed)
            seeds.append(seed)

        file = open(file_path, "a")

        # Loop through the zipped pairs
        for seed, score, loss in zip(seeds, qwk_score, test_loss):
            data += f"\t\tFor seed {seed}, score was {score} and test loss was {loss}\n"
        data += f"\tAverage QWK score: {np.average(qwk_score)} and Average Test Loss: {np.average(test_loss)} \n"
        file.write(data)

        # Close the file
        file.close()


def baseline(args, dataloaders_list, embedding_type, stride1, stride2, stride3, stride4, stride5, stride6, ks1, ks2, ks3):
    #print("VOCAB_SIZE: ", vocab_size)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline(args, stride1, stride2, stride3, stride4, stride5, stride6, ks1, ks2, ks3)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, dataloaders_list[0], dataloaders_list[1], args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score_for_seed, test_loss_for_seed =  evaluation.evaluate(model, dataloaders_list[2])
    print(f"Quadratic Weighted Kappa (QWK) Score on test set: {qwk_score_for_seed} and test loss is: {test_loss_for_seed}")
    return qwk_score_for_seed, test_loss_for_seed

if __name__ == "__main__":
    main()
    
