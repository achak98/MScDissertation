import os
import pandas as pd

def generate_bieo_labels(base_text, edus):
    labels = []
    words = base_text.split()
    edu_index = 0
    edu_words = edus[edu_index].split()

    for word in words:
        if word == edu_words[0]:
            if len(edu_words) == 1:
                labels.append("B")
                edu_index += 1
                if edu_index < len(edus):
                    edu_words = edus[edu_index].split()
            else:
                labels.append("B")
                edu_words = edu_words[1:]
        elif word in edu_words and edu_words[-1] != word:
            labels.append("I")
        elif word == edu_words[-1]:
            labels.append("E")
            edu_index += 1
            if edu_index < len(edus):
                edu_words = edus[edu_index].split()
        else:
            labels.append("O")

    return labels

def process_files(text_file, edu_file):
    with open(text_file, 'r') as f:
        base_text = f.read().strip()

    with open(edu_file, 'r') as f:
        edus = f.read().strip().split('\n')

    labels = generate_bieo_labels(base_text, edus)
    return base_text, labels

# Directory paths for text and EDU files
text_dir = 'path/to/text/files'
edu_dir = 'path/to/edu/files'

# Output CSV file path
output_csv = 'output.csv'

# Process files in the directory
data = []
for filename in os.listdir(text_dir):
    if filename.endswith('.txt'):
        text_file = os.path.join(text_dir, filename)
        edu_file = os.path.join(edu_dir, filename.replace('.txt', '.edu'))
        base_text, labels = process_files(text_file, edu_file)
        data.append([base_text, ' '.join(labels)])

# Create a DataFrame and save as CSV
df = pd.DataFrame(data, columns=['Text', 'Labels'])
df.to_csv(output_csv, index=False)

# Read the CSV file as a DataFrame
df = pd.read_csv(output_csv)

# Display the DataFrame
print(df)
