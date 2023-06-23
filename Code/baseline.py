import argsparser, dataloaders, evaluation, train, models
import random
import torch
import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler
import yielder

def main():

    embedding_types = ["w2v"] #["glove", "skipgram", "w2v"]
    file_path = "/home/achakravarty/Dissertation/Data/results/qwk.txt"
    args = argsparser.parse_args()
    gen = yielder.yield_hyps(args,embedding_types)
    seed = 42
    # Set the deterministic behavior for cudNN (if using GPUs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = ""
    for embedding_type, strides, kernels in gen:
        qwk_for_seeds = []
        t_loss_for_seeds = []
        seeds = []
        
        data += f"\nFor embedding type: {embedding_type} stride1: {strides[0]} ks1: {kernels[0]} stride2:{strides[1]} stride3: {strides[2]} ks2: {kernels[1]} stride4: {strides[3]} stride5: {strides[4]} ks3: {kernels[2]} stride6: {strides[5]} \n"
        for i in range(10):
            seed = random.randint(seed)
            # Set the random seed for Python's random module
            random.seed(seed)

            # Set the random seed for NumPy
            np.random.seed(seed)

            # Set the random seed for PyTorch
            torch.manual_seed(seed)

            # Set the random seed for CUDA operations (if using GPUs)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        

            qwk_score_over_folds = []
            test_loss_over_folds = []
            data += f"For seed:{seed} \n\n"
            k=10
            print("!!========================CREATING DATASET========================!!")
            data_set = dataloaders.create_datset(args, embedding_type)
            cross_val_loader = yielder.yield_crossval_dls(args=args, dataset=data_set,k_fold=k)
            for fold, dataloaders_list in enumerate(cross_val_loader):
                print(f"for fold: {fold+1}")
                # Parse the arguments
                qwk_score_for_fold, test_loss_for_fold = baseline(args, embedding_type, strides, kernels, dataloaders_list)
                qwk_score_over_folds.append(qwk_score_for_fold)
                test_loss_over_folds.append(test_loss_for_fold)
                data += f"\t\tFor fold {fold+1}, score was {qwk_score_for_fold} and test loss was {test_loss_for_fold}\n"
                file = open(file_path, "r")
                file.write(data)
                file.close()
            seeds.append(seed)
            data += f"\tOver folds, Average QWK score: {np.average(qwk_score_over_folds)} and Average Test Loss: {np.average(test_loss_over_folds)} \n"
            file = open(file_path, "r")
            file.write(data)
            file.close()
            qwk_for_seeds.append[np.average(qwk_score_over_folds)]
            t_loss_for_seeds.append[np.average(test_loss_over_folds)]
        data += f"\tOver all seeds and folds, Average QWK score: {np.average(qwk_for_seeds)} and Average Test Loss: {np.average(t_loss_for_seeds)} \n"
        file = open(file_path, "r")
        file.write(data)
        file.close()


def baseline(args, embedding_type, strides, kernels, dataloaders_list):
    #print("VOCAB_SIZE: ", vocab_size)
   # print("!!========================CREATING DATA LOADERS========================!!")
    #train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
   #     .create_data_loaders(args, embedding_type = embedding_type)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline(args, strides, kernels)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, dataloaders_list[0], dataloaders_list[1], args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score_for_seed, test_loss_for_seed =  evaluation.evaluate(model, dataloaders_list[2])
    print(f"Quadratic Weighted Kappa (QWK) Score on test set: {qwk_score_for_seed} and test loss is: {test_loss_for_seed}")
    return qwk_score_for_seed, test_loss_for_seed


if __name__ == "__main__":
    main()
    
