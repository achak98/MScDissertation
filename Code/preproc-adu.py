import pandas as pd
import os
from tqdm import tqdm
data_dir = "./../Data/pe"
essay_dir = data_dir
test_train_split_file = os.path.join(data_dir,"train-test-split.csv")
# Original kaggle training set
test_train_split = pd.read_csv(
    test_train_split_file, delimiter=";"
    )
headers = test_train_split.columns.tolist()
print(headers)
# Smaller training set used for this project
dataset = pd.DataFrame(
    {
        "ID": test_train_split["ID"],
        "SET": test_train_split["SET"]
    }
)
def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

adu_in = os.path.join(data_dir,"adu_in")
test = os.path.join(adu_in,"TEST")
train = os.path.join(adu_in,"TRAINING")
check_and_create_directory(adu_in)
check_and_create_directory(test)
check_and_create_directory(train)

for (eid,eset) in tqdm(zip(dataset["ID"], dataset["SET"])):
    annfile = os.path.join(data_dir,str(eid)+".ann")
    txtfile = os.path.join(data_dir,str(eid)+".txt")
    data = ""
    if eset == "TRAIN":
        ann_output_file_name = os.path.join(train,f"{eid}.out.edus")
        text_output_file_name = os.path.join(train,f"{eid}.out")
    elif eset == "TEST":
        ann_output_file_name = os.path.join(test,f"{eid}.out.edus")
        text_output_file_name = os.path.join(test,f"{eid}.out")
    sorted_t_anns = []
    with open(txtfile, "r") as text:
        essay = text.read()
        with open(annfile, "r") as ann:
            annotations = ann.read().split("\n")
            t_anns = []
            for annotation in annotations:
                if annotation.startswith("T"):
                    print(annotation)
                    first_split = annotation.split("\t") 
                    second_split = first_split[1].split()
                    t_anns.append([second_split[1],second_split[2]])
            sorted_t_anns = sorted(t_anns, key=lambda x: int(x[0]))
        data = ""
        for (start, end) in sorted_t_anns:
            data += essay[int(start):int(end)] + "\n"
        modified_essay = essay.replace(". ", ".\n")
        with open(ann_output_file_name, "w") as ann_op:
            ann_op.write(data)
        with open(text_output_file_name, "w") as txt_op:
            txt_op.write(modified_essay)
