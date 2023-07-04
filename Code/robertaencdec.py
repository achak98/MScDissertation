import argsparser, dataloaders, evaluation, train, models
import random
import torch
import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler
import yielder
import pandas as pd
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import logging
from transformers import logging as hf_logging

# Set the verbosity level to 'error' to suppress warnings
hf_logging.set_verbosity(logging.ERROR)

def normalise_scores(score, prompt):
    if prompt == 1:
        pass
        score = (score-2)/10
    elif prompt == 2:
        score = (score-2)/8
    elif prompt == 3:
        score = score/3
    elif prompt == 4:
        score = score/3
    elif prompt == 5:
        score = score/4
    elif prompt == 6:
        score = score/4
    elif prompt == 7:
        score = score/30
    elif prompt == 8:
        score = score/60
    return score

class EssayDataset(Dataset):
    def __init__(self,tsv_file, max_chunk_length, prompt):
        df = pd.read_csv(tsv_file, sep='\t', encoding='ISO-8859-1')
        self.data = df[df.iloc[:, 1] == prompt]
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_chunk_length = max_chunk_length
        self.prompt = prompt
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        essay = self.data['essay'][index]
        score = self.data['domain1_score'][index]
        score = normalise_scores(score=score, prompt=self.prompt)
        encoded_input = self.tokenizer.encode_plus(essay, add_special_tokens=True, truncation=True, max_length=self.max_chunk_length, padding='max_length')
        return ({
            'input_ids': encoded_input['input_ids'],
            'attention_mask': encoded_input['attention_mask'],
            'score': score
        })

    """
    def _split_into_chunks(self, essay):
        chunks = []
        chunk_start = 0

        while chunk_start < len(essay):
            chunk_end = min(chunk_start + self.max_chunk_length, len(essay))
            chunk = essay[chunk_start:chunk_end]
            chunks.append(chunk)
            chunk_start = chunk_start + self.max_chunk_length

        return chunks
    """
def main():
    max_chunk_length = 512
    file_path = "./../Data/results/biroberta/qwk"
    args = argsparser.parse_args()
    seed = 0
    # Set the deterministic behavior for cudNN (if using GPUs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = ""
   
    qwk_for_seeds = []
    t_loss_for_seeds = []
    seeds = []
    for prompt in range(1,9):
        file_path += f"_{prompt}.txt"
        for i in range(10):
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

            qwk_score_over_folds = []
            test_loss_over_folds = []
            data += f"For seed:{seed} \n\n"
            k=10
            print("!!========================CREATING DATASET========================!!")
            data_set = EssayDataset(args.dataDir + '/training_set_rel3.tsv', max_chunk_length, prompt)
            cross_val_loader = yielder.yield_crossval_dls_roberta(args=args, dataset=data_set,k_fold=k)
            for fold, dataloaders_list in enumerate(cross_val_loader):
                print(f"for fold: {fold+1}")
                # Parse the arguments
                qwk_score_for_fold, test_loss_for_fold = robertaencdec(args, dataloaders_list, prompt)
                qwk_score_over_folds.append(qwk_score_for_fold)
                test_loss_over_folds.append(test_loss_for_fold)
                data += f"\t\tFor fold {fold+1}, score was {qwk_score_for_fold} and test loss was {test_loss_for_fold}\n"
                file = open(file_path, "w")
                file.write(data)
                file.close()
            seeds.append(seed)
            data += f"\tOver folds, Average QWK score: {np.average(qwk_score_over_folds)} and Average Test Loss: {np.average(test_loss_over_folds)} \n"
            file = open(file_path, "w")
            file.write(data)
            file.close()
            qwk_for_seeds.append[np.average(qwk_score_over_folds)]
            t_loss_for_seeds.append[np.average(test_loss_over_folds)]
        data += f"\tOver all seeds and folds, Average QWK score: {np.average(qwk_for_seeds)} and Average Test Loss: {np.average(t_loss_for_seeds)} \n"
        file = open(file_path, "w")
        file.write(data)
        file.close()

def robertaencdec(args, dataloaders_list, prompt):
    
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.RobertaEncDec()
    #Roberta
    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train_roberta(model, dataloaders_list[0], dataloaders_list[1], args.num_epochs, args.lr, prompt)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score_for_seed, test_loss_for_seed =  evaluation.evaluate(model, dataloaders_list[2], prompt)
    print(f"Quadratic Weighted Kappa (QWK) Score on test set: {qwk_score_for_seed} and test loss is: {test_loss_for_seed}")
    return qwk_score_for_seed, test_loss_for_seed
    
if __name__ == "__main__":
    main()



