import os, sys, json
import nltk
from tqdm import tqdm
nltk.download('punkt')

import random
random.seed(0)


def get_vocab(train_file,out_file):
    f = open(train_file,'r')
    lines = [x.strip() for x in f.readlines()]
    print(len(lines))
    
    vocab_dict = {}
    for line in tqdm(lines):
        words = nltk.word_tokenize(line)
        for word in words:
            if word not in vocab_dict.keys():
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
        
    print(len(list(vocab_dict.keys()))) #23217
    with open(out_file, 'w') as f:
        json.dump(vocab_dict,f,indent=4)
    f.close()


train_file = 'train_data/CHILDES_train_ordered.txt'
out_file = 'train_data/CHILDES_train_ordered_vocab.json'

get_vocab(train_file,out_file) #68128 unique words