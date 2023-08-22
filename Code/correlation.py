import pandas as pd
import numpy as np
import os
from transformers import  AutoTokenizer
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

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

def count_words(text):
    words = text.split()
    return len(words)

for (default_essay,essay_id,essay_set, score) in tqdm(zip(dataset['essay'], dataset["essay_id"], dataset["essay_set"], dataset["score"]), total=len(dataset['essay'])):
    no_of_adus = 0
    no_of_edus = 0
   
    wc = count_words(default_essay)
   
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
    data.append([essay_set, score, no_of_edus, no_of_adus, wc])
df = pd.DataFrame(data, columns=['essay_set', 'score', 'edu_count', 'ac_count', 'word_count'])
print(df.head)

for essay_set in range (1,9):
    scaler = StandardScaler()
    scaled = []
    for essay_set in range(1, 9):
        score = df[df["essay_set"] == essay_set]["score"].to_frame()
        s = scaler.fit_transform(score).reshape(-1)
        scaled = np.append(scaled, s)

    scaled_df = df.copy()
    scaled_df["scaled_score"] = scaled

print(scaled_df.head)

col1 = scaled_df['scaled_score']
col2 = scaled_df['ac_count']
col3 = scaled_df['edu_count']
col4 = scaled_df['word_count']

corr_coefficient, p_value = pearsonr(col1, col2)
print(f"Correlation Coefficient between scaled_score and ac_count: {corr_coefficient}")
print(f"P-value: {p_value}")

corr_coefficient, p_value = pearsonr(col1, col3)
print(f"Correlation Coefficient between scaled_score and edu_count: {corr_coefficient}")
print(f"P-value: {p_value}")

corr_coefficient, p_value = pearsonr(col1, col4)
print(f"Correlation Coefficient between scaled_score and word_count: {corr_coefficient}")
print(f"P-value: {p_value}")

corr_coefficient, p_value = pearsonr(col2, col3)
print(f"Correlation Coefficient between ac_count and edu_count: {corr_coefficient}")
print(f"P-value: {p_value}")

corr_coefficient, p_value = pearsonr(col2, col4)
print(f"Correlation Coefficient between ac_count and word_count: {corr_coefficient}")
print(f"P-value: {p_value}")

corr_coefficient, p_value = pearsonr(col3, col4)
print(f"Correlation Coefficient between edu_count and word_count: {corr_coefficient}")
print(f"P-value: {p_value}")

