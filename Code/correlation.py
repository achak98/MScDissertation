import pandas as pd
import numpy as np
import os
from transformers import  AutoTokenizer
from tqdm.auto import tqdm

data_dir = "./../Data/ASAP-AES/"
# Original kaggle training set
kaggle_dataset = pd.read_csv(
    os.path.join(data_dir,"training_set_rel3.tsv"), sep="\t", encoding="ISO-8859-1"
)
# Smaller training set used for this project
dataset = pd.DataFrame(
    {
        "essay_id": kaggle_dataset["essay_id"],
        "essay_set": kaggle_dataset["essay_set"],
        "essay": kaggle_dataset["essay"],
        "rater1": kaggle_dataset["rater1_domain1"],
        "rater2": kaggle_dataset["rater2_domain1"],
        "score": kaggle_dataset["domain1_score"],
    }
)

#tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
#tokenizer.add_special_tokens({"additional_special_tokens": [special_token_adu, special_token_prompt]})

edu_dir = os.path.join(data_dir,"seg-edu")
adu_dir = os.path.join(data_dir,"seg-adu")

data = []

for (default_essay,essay_id,essay_set, score) in tqdm(zip(dataset['essay'], dataset["essay_id"], dataset["essay_set"], dataset["score"]), total=len(dataset['essay'])):
    no_of_adus = 0
    no_of_edus = 0
    if  os.path.exists(os.path.join(adu_dir, str(essay_id) + ".out")):
        with open(os.path.join(adu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                #essay += f"{special_token_adu} {line.strip()} "
                no_of_adus+=1
    else:
       no_of_adus=1
    if  os.path.exists(os.path.join(edu_dir, str(essay_id) + ".out")):
        with open(os.path.join(edu_dir, str(essay_id) + ".out"), "r") as file:
            for line in file:
                no_of_edus+=1
    else:
        no_of_edus = 1
    data.append([score, no_of_edus, no_of_adus])
    df = pd.DataFrame(data, columns=['Score', 'EDU Count', 'ADU Count'])
    print(df.head)
