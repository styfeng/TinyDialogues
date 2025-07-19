import os, sys, json
import nltk
import csv
from tqdm import tqdm
nltk.download('punkt')

import random
random.seed(0)


def read_dict_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            dictionary = json.loads(line)
            data.append(dictionary)
    return data


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)
        

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        for dictionary in data:
            json.dump(dictionary, file)
            file.write('\n')
    file.close()


def write_txt_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")


def write_csv_file(file_path, list_data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in list_data:
            writer.writerow(item)


def write_tabs_to_csv_file(file_path, list_data):
    with open(file_path, 'w', newline='') as file:
        for item in list_data:
            file.write('\t'.join(item) + '\n')


def filter_examples(in_file,vocab_dict):
    input_examples = read_csv_file(in_file)

    original_len = len(input_examples)
    output_examples = []
    total_word_count = 0
    unknown_word_count = 0
    
    for example in tqdm(input_examples):
        
        if 'SimVerb-3500' in in_file:
            word1 = example[0].split('\t')[0]
            word2 = example[0].split('\t')[1]
        
        elif 'MTest-3000' in in_file:
            word1 = example[0].split(' ')[0]
            word2 = example[0].split(' ')[1]
        
        else:
            word1 = example[0].split(';')[0]
            word2 = example[0].split(';')[1]

        include_example = True
        total_word_count += 2
        
        for word in [word1,word2]:
            if word not in vocab_dict.keys():
                include_example = False
                unknown_word_count += 1
                #continue
            #elif vocab_dict[word] < 2:
            #    include_example = False
            #    continue
        
        if include_example:
            output_examples.append(example)
            
    print(len(output_examples))
    print(f"{len(output_examples)} out of {len(input_examples)} ({len(output_examples)/len(input_examples)} fraction) kept")
    print(f"{unknown_word_count} out of {total_word_count} ({unknown_word_count/total_word_count} fraction) words unknown")
    
    out_file = in_file.replace('original','')
    if 'SimVerb-3500' in in_file:
        write_tabs_to_csv_file(out_file,output_examples)
    else:
        write_csv_file(out_file,output_examples)


vocab_file = 'train_data/CHILDES_train_ordered_vocab.json'
vocab_dict = read_dict_file(vocab_file)

in_path = '../LexiContrastiveGrd/src/llm_devo/word_sim/data/original'
in_files = os.listdir(in_path)
in_files = [os.path.join(in_path,x) for x in in_files]

for in_file in in_files:
    print(f'\n{in_file}')
    filter_examples(in_file,vocab_dict)