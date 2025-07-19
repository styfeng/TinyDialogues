import pdb
import argparse
import re
import copy
import numpy as np
from tqdm import tqdm
import os
import csv
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG

Wordbank_DATA_PATH = os.path.join(
        './data', 'wordbank_trajs/wordbank_item_ws_ea.csv')
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ_ORG, 'dist_sent_results/fast_len_lt30_roberta-large_clm_opt-125m')


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--source',
            default='yourdict', type=str, action='store')
    parser.add_argument(
            '--num_words_per_cat_per_end',
            default=1, type=int, action='store')
    parser.add_argument(
            '--groups_of_interest',
            default='vehicles,animals,toys,food_drink,clothing,body_parts,household,furniture_rooms,outside,places', 
            type=str, action='store')
    return parser


class WordBankReader:
    def __init__(
            self, data_path, 
            month_range=list(range(16, 31))):
        self.data_path = data_path
        self.month_range = month_range
        self.get_data()
        self.get_all_categories()
        self.get_words_by_cat()
        self.get_auc_per_word()

    def get_data(self):
        all_rows = []
        with open(self.data_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_rows.append(row)
        self.all_rows = all_rows

    def get_all_categories(self):
        all_categories = []
        for row in self.all_rows:
            category = row['category']
            if category not in all_categories:
                all_categories.append(category)
        self.all_categories = all_categories

    def clean_raw_word(self, raw_word):
        if '(' in raw_word:
            raw_word = raw_word.split('(')[0]
        if ' ' in raw_word:
            return None
        if '*' in raw_word:
            return None
        return raw_word

    def get_words_by_cat(self):
        words_by_cat = dict()
        for row in self.all_rows:
            category = row['category']
            if category not in words_by_cat:
                words_by_cat[category] = []
            raw_word = row['item_definition']
            words_by_cat[category].append(raw_word)
        self.words_by_cat = words_by_cat

    def get_auc_per_word(self):
        auc_per_word = {}
        for row in self.all_rows:
            raw_word = row['item_definition']
            auc_per_word[raw_word] = sum(
                    [float(row[str(mth)]) for mth in self.month_range])
        self.auc_per_word = auc_per_word

    def register_finished_words(
            self, result_dir):
        all_files = os.listdir(result_dir)
        all_files = filter(lambda x: x.endswith('.pkl'), all_files)
        all_files = sorted(all_files)
        num_of_files_by_target_words = dict()
        for _file in all_files:
            target_word = _file.split('_')[0]
            if target_word not in num_of_files_by_target_words:
                num_of_files_by_target_words[target_word] = 0
            num_of_files_by_target_words[target_word] += 1
        self.finished_words = list(num_of_files_by_target_words.keys())

    def get_clean_words_by_cat(self, cat):
        words = self.words_by_cat[cat]
        words = list(filter(lambda x: self.clean_raw_word(x) is not None, words))
        words = sorted(words)
        return words

    def output_next_words(
            self, num_words_per_cat_per_end=1,
            groups_of_interest='vehicles,animals,toys,food_drink,clothing,body_parts,household,furniture_rooms,outside,places',
            ):
        next_words = []
        groups_of_interest = groups_of_interest.split(',')
        for group in groups_of_interest:
            words = self.words_by_cat[group]
            words = list(filter(lambda x: x not in self.finished_words, words))
            words = list(filter(lambda x: self.clean_raw_word(x) is not None, words))
            words = list(sorted(words, key=lambda x: self.auc_per_word[x]))
            if num_words_per_cat_per_end * 2 >= len(words):
                next_words.extend(words)
            else:
                next_words.extend(words[:num_words_per_cat_per_end])
                next_words.extend(words[-num_words_per_cat_per_end:])
        output_str = ' '.join([f'"{self.clean_raw_word(word)}"' for word in next_words])
        print(output_str)
        print(len(next_words))


def get_wdbank_adjs():
    reader = WordBankReader(Wordbank_DATA_PATH)
    adj_words = reader.get_clean_words_by_cat('descriptive_words')
    return adj_words


def get_wdbank_verbs():
    reader = WordBankReader(Wordbank_DATA_PATH)
    adj_words = reader.get_clean_words_by_cat('action_words')
    return adj_words


def get_wdbank_nouns():
    reader = WordBankReader(Wordbank_DATA_PATH)
    noun_cats = 'vehicles,animals,toys,food_drink,clothing,body_parts,household,furniture_rooms,outside,places'.split(',')
    noun_words = []
    for _cat in noun_cats:
        noun_words.extend(reader.get_clean_words_by_cat(_cat))
    return noun_words


def main():
    parser = get_parser()
    args = parser.parse_args()

    wordbank_reader = WordBankReader(
            Wordbank_DATA_PATH)
    wordbank_reader.register_finished_words(
            os.path.join(RESULT_DIR, args.source))
    print('Finished words: ', wordbank_reader.finished_words)

    wordbank_reader.output_next_words(
            num_words_per_cat_per_end=args.num_words_per_cat_per_end,
            groups_of_interest=args.groups_of_interest)


if __name__ == '__main__':
    main()
