import pandas as pd
import os
from tqdm import tqdm
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
def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

check_and_create_directory(os.path.join(data_dir,"edu_in"))

for (essay,essay_id) in tqdm(zip(dataset["essay"], dataset["essay_id"])):
    modified_essay = essay.replace(". ", ".\n")
    output_file_name = os.path.join(data_dir,"edu_in",f"{essay_id}.out")
    with open(output_file_name, "w") as file:
        file.write(modified_essay)