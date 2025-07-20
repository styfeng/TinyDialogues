import os
import sys
import re

train_or_val = sys.argv[1]
n = int(sys.argv[2])  # Number of times to repeat the lines in each file

def replace_speaker_labels(line):
    """Replace 'Toddler' and 'Teenager' (and other variants based on the speaker label analyses) with 'Child'."""
    line = line.strip().replace("**Toddler**", "**Child**").replace("**5-Year-Old Child**","**Child**").replace("**5-year-old Child**","**Child**").replace("**5-Year-Old**","**Child**").replace("**5-year-old**","**Child**").replace("**5-year-old child**","**Child**").replace("**10-Year-Old Child**","**Child**").replace("**10-year-old Child**","**Child**").replace("**10-Year-Old**","**Child**").replace("**10-year-old**","**Child**").replace("**10-year-old child**","**Child**").replace("**Teenager**", "**Child**").replace("**15-Year-Old Teenager**", "**Child**").replace("**Teenager (Alex)**", "**Child**").replace("**Alex**", "**Child**").replace("**Teen**", "**Child**").replace("**Teenager (Chris)**", "**Child**").replace("**Teenager (Sam)**", "**Child**") + '\n'
    # Get rid of extra newline tokens at beginning of lines
    if line.startswith("\\n"):
        line = line[2:]
    # Ensure all utterance separator tokens are double newline tokens with whitespace before and after
    line = line.replace("\\n'\\n**", ' \\n\\n **').replace('\\n**',' \\n\\n **')
    return line

def read_and_process_file(file_path, n):
    """Read a file, repeat lines n times, and replace speaker labels."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    processed_lines = [replace_speaker_labels(line) for line in lines]
    return processed_lines * n

def combine_files(file_list, n, reverse=False):
    """Combine files in given or reverse order, processing each file's lines."""
    combined_lines = []
    if reverse:
        file_list = file_list[::-1]
    
    for file_path in file_list:
        combined_lines.extend(read_and_process_file(file_path, n))
    
    return combined_lines

def write_to_file(file_path, lines):
    """Write combined lines to the output file."""
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Input list of file names
if train_or_val == 'train':
    input_files = ['data/train_data/tinydialogue_age-2_combined_dialogues_endoftext_train.txt',\
                   'data/train_data/tinydialogue_age-5_combined_dialogues_endoftext_train.txt',\
                   'data/train_data/tinydialogue_age-10_combined_dialogues_endoftext_train.txt',\
                   'data/train_data/tinydialogue_age-15_combined_dialogues_endoftext_train.txt'] #these files can be found at https://huggingface.co/datasets/styfeng/TinyDialogues/blob/main/individual_age_data.zip

    # Combine files in given order
    combined_lines_order = combine_files(input_files, n)
    write_to_file(f'data/train_data/tinydialogue_train_ordered_{n}n.txt', combined_lines_order)

    # Combine files in reverse order
    combined_lines_reverse = combine_files(input_files, n, reverse=True)
    write_to_file(f'data/train_data/tinydialogue_train_reversed_{n}n.txt', combined_lines_reverse)

if train_or_val == 'val':
    input_files = ['data/train_data/tinydialogue_age-2_combined_dialogues_endoftext_val.txt',\
                   'data/train_data/tinydialogue_age-5_combined_dialogues_endoftext_val.txt',\
                   'data/train_data/tinydialogue_age-10_combined_dialogues_endoftext_val.txt',\
                   'data/train_data/tinydialogue_age-15_combined_dialogues_endoftext_val.txt'] #these files can be found at https://huggingface.co/datasets/styfeng/TinyDialogues/blob/main/individual_age_data.zip

    # Combine files in given order
    combined_lines_order = combine_files(input_files, n)
    write_to_file(f'data/train_data/tinydialogue_val_ordered_{n}n.txt', combined_lines_order)

    # Combine files in reverse order
    combined_lines_reverse = combine_files(input_files, n, reverse=True)
    write_to_file(f'data/train_data/tinydialogue_val_reversed_{n}n.txt', combined_lines_reverse)

print("Files have been processed and written successfully.")