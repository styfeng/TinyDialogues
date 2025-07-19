import os, sys, json
import nltk
from tqdm import tqdm
nltk.download('punkt')

import random
random.seed(0)

speaker_label = sys.argv[1]


def read_dict_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            dictionary = json.loads(line)
            data.append(dictionary)
    return data


def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        for dictionary in data:
            json.dump(dictionary, file)
            file.write('\n')
    file.close()


def convert_examples(in_file, speaker_label):
    input_examples = read_json_file(in_file)
    original_len = len(input_examples)
    output_examples = []
    
    for example in tqdm(input_examples):
        
        sentence1 = example['sentence_good']
        sentence2 = example['sentence_bad']
        sentence1_dialogue = f'\\n\\n **{speaker_label}**: ' + sentence1 + ' \\n\\n'
        sentence2_dialogue = f'\\n\\n **{speaker_label}**: ' + sentence2 + ' \\n\\n'
        
        example['sentence_good'] = sentence1_dialogue
        example['sentence_bad'] = sentence2_dialogue

        output_examples.append(example)

    out_file = in_file.replace('filter-data_zorro',f'filter-data_zorro_dialogue-format-CHILDES_{speaker_label}')
    write_json_file(out_file,output_examples)


in_path = 'evaluation-pipeline/filter-data_zorro'
in_files = os.listdir(in_path)
in_files = [os.path.join(in_path,x) for x in in_files]

for in_file in in_files:
    print(f'\n{in_file}')
    convert_examples(in_file, speaker_label)