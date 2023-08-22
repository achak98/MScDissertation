import pandas as pd
import numpy as np
import os
from transformers import  AutoTokenizer
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


# Example usage:
save_directory = "./../Data/results/corr"
check_and_create_directory(save_directory)

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
    len_str = len(default_essay)
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
    data.append([essay_set, score, no_of_edus, no_of_adus, wc, len_str])
df = pd.DataFrame(data, columns=['essay_set', 'score', 'edu_count', 'ac_count', 'word_count', "len_str"])
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

correlation_matrix = {}

columns=['essay_set', 'score', 'edu_count', 'ac_count', 'word_count', "len_str", "scaled_score"]

# Calculate Pearson correlation coefficients and p-values for all possible pairs
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        col1 = scaled_df[columns[i]]
        col2 = scaled_df[columns[j]]
        
        corr_coeff, p_value = pearsonr(col1, col2)
        correlation_matrix[f"{columns[i]} vs {columns[j]}"] = {"Correlation": corr_coeff, "P-value": p_value}

# Print the correlation matrix
data = ""
for pair, values in correlation_matrix.items():
    print(pair)
    print("Correlation Coefficient:", values["Correlation"])
    print("P-value:", values["P-value"])
    print()
    data += f"""{pair}
                "Correlation Coefficient:", {values["Correlation"]}
                "P-value:", {values["P-value"]}\n"""

    file = open(os.path.join(save_directory, f"corr.txt"), "w")
    file.write(data)
    file.close()