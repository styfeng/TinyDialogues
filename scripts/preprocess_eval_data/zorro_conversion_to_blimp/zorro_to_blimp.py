import os
import json

# Define the root folder containing the .txt files in subfolders
root_folder = 'evaluation-pipeline/zorro_data'  # Change this to the correct root folder
output_folder = 'evaluation-pipeline/filter-data_zorro'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process the lines and format the sentences
def format_sentence(line):
    return line.strip().replace(" .", ".").replace(" ?", "?")

# Loop through all subfolders
for subdir, _, files in os.walk(root_folder):
    if not files:
        continue
    subfolder_name = os.path.basename(subdir)
    output_file_path = os.path.join(output_folder, f"{subfolder_name}.json")
    
    # Open the output jsonl file
    with open(output_file_path, 'w') as output_file:
        # Loop through all .txt files in the subfolder
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(subdir, filename)
                
                # Read and process the .txt file
                with open(file_path, 'r') as f:
                    lines = [format_sentence(line) for line in f.readlines()]
                    
                    # Combine pairs of lines into a single jsonl line
                    for i in range(0, len(lines), 2):
                        if i+1 < len(lines):
                            sentence_bad = lines[i]
                            sentence_good = lines[i+1]
                            example = {
                                "sentence_good": sentence_good,
                                "sentence_bad": sentence_bad,
                                "phenomena": subfolder_name
                            }
                            output_file.write(json.dumps(example) + '\n')

print("All files have been processed and written to the output folder.")