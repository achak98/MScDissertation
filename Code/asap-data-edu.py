import pandas as pd
import os
from tqdm import tqdm
kaggle_dataset = pd.read_csv(
    "./../Data//ASAP-AES/training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1"
)
print(kaggle_dataset)
print(type(kaggle_dataset))
output_dir = "./../Data//ASAP-AES/edu"
os.makedirs(output_dir, exist_ok=True)
for index, row in tqdm(kaggle_dataset.iterrows()):
    print(row)
    file_name = str(row.loc['essay_id'])+".out"
    essay_data = row.loc['essay']
    # Open the file in write mode
    with open(os.path.join(output_dir, file_name), "w") as file:
        # Loop through each epoch's data and write it to the file
        file.write(essay_data)