import torch
import ipdb
import pandas as pd
import copy
import argparse
import functools
import os
import re
import numpy as np
import pickle
import csv
from tqdm import tqdm
from scipy.stats import spearmanr

import llm_devo.analysis.use_lm_eval as use_lm_eval
from llm_devo.env_vars import ROOT_DIR, DEBUG, ROOT_DIR_FREQ_ORG

#BELOW RESULT_DIR CODE CHANGED BY STEVEN

'''
RESULT_DIR = os.path.join(
        #ROOT_DIR,
        ROOT_DIR_FREQ_ORG,
        'llm_devo_word_sim_results')
'''

current_directory = os.getcwd()
print(current_directory)

RESULT_DIR = os.path.join(
        current_directory,
        'llm_devo_word_sim_results')


def cosine_similarity(a, b):
    return ((np.dot(a, b)) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))))


def check_and_create_csv(file_name):
    # Check if the file exists
    if not os.path.isfile(file_name):
        # If the file does not exist, create it with the default header
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Model", "Eval Dataset", "Max Layer Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")


def append_row_to_csv(file_name, model_path, input_str, best_layer_score):
    # Format the final average score as a percentage with two decimal places
    #final_avg_score_formatted = f"{final_avg_score:.2f}"
    
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the new row
        writer.writerow([model_path, input_str, best_layer_score])
    print(f"Appended row to '{file_name}'.")


def check_and_create_final_avg_csv(file_name):
    # Check if the file exists
    if not os.path.isfile(file_name):
        # If the file does not exist, create it with the default header
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Model", "Final Avg Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")


def append_row_to_final_avg_csv(file_name, model_path, final_avg_score):
    # Format the final average score as a percentage with two decimal places
    final_avg_score_formatted = f"{final_avg_score:.2f}"
    
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the new row
        writer.writerow([model_path, final_avg_score_formatted])
    print(f"Appended row to '{file_name}'.")
    

def get_parser():
    parser = argparse.ArgumentParser(
            description='Get evaluation metrics through lm_eval')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--output_file', default=None, type=str, action='store')
    parser.add_argument(
            '--output_csv', default=None, type=str, action='store')
    parser.add_argument(
            '--output_final_avg_score_csv', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--pretrained', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--high_level_task', default='human_sim', 
            type=str, action='store')
    parser.add_argument(
            '--sim_metric', default='cosine', 
            type=str, action='store')
    parser.add_argument(
            '--extra_forward_mode', default=None, 
            type=str, action='store')
    parser.add_argument(
            '--which_ckpt', default=None, 
            type=str, action='store')
    #args = parser.parse_args()
    return parser


HUMAN_CSVs = [
        'rg65', 'simlex999', 'wordsim353', 'SimVerb-3500',
        'MTest-3000',
        ]
def load_human_sim_data(
        data_folder='data',
        csv_files=HUMAN_CSVs,
        ):
    string_lists_sims = dict()
    for csv_file in csv_files:
        file_path = os.path.join(
                data_folder, f'{csv_file}.csv')
        sep = ';'
        if csv_file == 'SimVerb-3500':
            sep = '\t'
        if csv_file == 'MTest-3000':
            sep = ' '
        intrinsic_evaluation = pd.read_csv(
                file_path, index_col=None,
                sep=sep, header=None)
        rel_idx = 2
        if csv_file == 'SimVerb-3500':
            rel_idx = 3
        words_left= intrinsic_evaluation[0].tolist()
        words_right = intrinsic_evaluation[1].tolist()
        relatedness = intrinsic_evaluation[rel_idx].tolist()
        string_lists_sims[csv_file] = dict(
                words_left=words_left,
                words_right=words_right,
                relatedness=relatedness,
                )
    return string_lists_sims


class WordSimRunner(use_lm_eval.LMEvalRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.args = args
        self.result_dir = result_dir
        self.batch_size = 16
        self.hidden_states_pos = 2
        self.has_eos_token = False
        self.load_string_lists_sims()
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()
        self.update_has_eos_token()

    def get_task_in_res(self):
        task_in_res = self.args.high_level_task
        if getattr(self.args, 'sim_metric', 'cosine') != 'cosine':
            task_in_res += f'_mtc_{self.args.sim_metric}'
        return task_in_res

    def load_human_string_lists_sims(self):
        self.string_lists_sims = load_human_sim_data()
        self.datasets = copy.copy(HUMAN_CSVs)

    def load_string_lists_sims(self):
        high_level_task = self.args.high_level_task
        if high_level_task == 'human_sim':
            self.load_human_string_lists_sims()
        else:
            raise NotImplementedError

    def get_all_model_inputs(self, string_list):
        tokenizer = self.lm.tokenizer
        all_input_ids = []
        all_attention_masks = []
        for string in string_list:
            input_str = [string]
            if self.add_top_blank:
                input_str = [' ' + string]
            if self.args.pretrained == 'gpt2':
                input_str = [tokenizer.bos_token + input_str[0]]
            inputs = tokenizer(
                    input_str, return_tensors="pt",
                    add_special_tokens=True)
            all_input_ids.append(inputs.input_ids)
            if 'attention_mask' in inputs:
                all_attention_masks.append(inputs.attention_mask)
        return all_input_ids, all_attention_masks

    def get_all_embeddings(
            self, input_ids, attention_masks,
            additional_forward_params=None):
        model = getattr(self.lm, 'gpt2', None) # the legacy name for the loaded model
        if model is None:
            model = self.lm.model
        single_extra_forward_kwargs = getattr(
                self.lm, 'extra_forward_kwargs', {})
        input_lens = [_id.shape[1] for _id in input_ids]
        input_lens = np.asarray(input_lens)
        sorted_idxs = np.argsort(input_lens)

        returned_embeddings = []
        diff_lens = np.unique(input_lens)
        all_idxs = []
        for _len in diff_lens:
            now_idx = np.where(input_lens == _len)[0]
            all_idxs.append(now_idx)
            new_input_ids = torch.cat([input_ids[_idx] for _idx in now_idx])
            if len(attention_masks) > 0:
                new_att_masks = torch.cat([attention_masks[_idx] for _idx in now_idx])
            else:
                new_att_masks = None

            for sta_idx in range(0, len(now_idx), self.batch_size):
                end_idx = min(sta_idx + self.batch_size, len(now_idx))
                _input_ids = new_input_ids[sta_idx : end_idx]
                if new_att_masks is not None:
                    _att_masks = new_att_masks[sta_idx : end_idx]

                extra_forward_kwargs = copy.copy(single_extra_forward_kwargs)
                if additional_forward_params is not None:
                    for key, value in additional_forward_params.items():
                        now_value = [
                                value[_idx]
                                for _idx in now_idx[sta_idx : end_idx]]
                        now_value = torch.stack(now_value, dim=0).to(self.lm._device)
                        extra_forward_kwargs[key] = now_value

                for key in extra_forward_kwargs:
                    if extra_forward_kwargs[key].size(0) != end_idx - sta_idx:
                        now_ts = extra_forward_kwargs[key]
                        rep_sizes = [1] * now_ts.ndim
                        rep_sizes[0] = end_idx - sta_idx
                        extra_forward_kwargs[key] = now_ts.repeat(*rep_sizes)
                
                with torch.no_grad():
                    if new_att_masks is not None:
                        text_features = model(
                                input_ids=_input_ids.to(self.lm._device), 
                                attention_mask=_att_masks.to(self.lm._device),
                                output_hidden_states=True,
                                **extra_forward_kwargs)
                    else:
                        text_features = model(
                                input_ids=_input_ids.to(self.lm._device), 
                                output_hidden_states=True,
                                **extra_forward_kwargs)

                if 'hidden_states' in text_features:
                    hidden_states = text_features['hidden_states']
                elif 'multimodal_output' in text_features: # special case for Flava models
                    hidden_states = []
                    for _state in text_features['text_output']['hidden_states']:
                        hidden_states.append(_state)
                    num_txt_tks = text_features['text_embeddings'].shape[1]
                    for _state in text_features['multimodal_output']['hidden_states']:
                        hidden_states.append(_state[:, -num_txt_tks:, :])
                elif 'text_output' in text_features: # special case for Flava models
                    hidden_states = text_features['text_output']['hidden_states']
                else:
                    hidden_states = text_features[self.hidden_states_pos]
                embeddings = []
                for layer_vectors in hidden_states:
                    if not isinstance(layer_vectors, np.ndarray):
                        layer_vectors = layer_vectors.cpu().numpy()
                    representation = layer_vectors[:, self.extraction_id]
                    embeddings.append(representation)
                returned_embeddings.append(embeddings)
        all_idxs = np.concatenate(all_idxs)
        no_layers = len(returned_embeddings[0])
        new_returned_embeddings = []
        for idx in range(no_layers):
            _embds = np.concatenate(
                    [returned_embeddings[inner_idx][idx]
                    for inner_idx in range(len(returned_embeddings))])
            new_embds = np.zeros_like(_embds)
            for curr_idx, new_idx in enumerate(all_idxs):
                new_embds[new_idx] = _embds[curr_idx]
            new_returned_embeddings.append(new_embds)
        return new_returned_embeddings

    def get_embedding_list(
            self, string_list,
            has_eos_token=False,
            add_top_blank=False):
        if (self.args.pretrained is not None)\
                and self.args.pretrained.startswith('glove_d'):
            embs = []
            for word in string_list:
                if word not in self.lm:
                    print(word, ' Out of Glove list!')
                    embs.append(np.zeros_like(embs[-1]))
                else:
                    embs.append(self.lm[word])
            return (np.asarray(embs),)

        extraction_id = -2
        if not has_eos_token:
            extraction_id = -1

        returned_embeddings = []
        self.add_top_blank = add_top_blank
        self.extraction_id = extraction_id

        input_ids, attention_masks = self.get_all_model_inputs(string_list)
        returned_embeddings = self.get_all_embeddings(
                input_ids, attention_masks)
        return returned_embeddings

    def get_layerwise_scores_for_one_model_one_ds(
            self, which_ds, add_top_blank):

        words_left = self.string_lists_sims[which_ds]['words_left']
        words_right = self.string_lists_sims[which_ds]['words_right']
        relatedness = self.string_lists_sims[which_ds]['relatedness']

        embd_list_kwargs = dict(
                has_eos_token=self.has_eos_token,
                add_top_blank=add_top_blank,
                )
        words_left_embeddings = self.get_embedding_list(words_left, **embd_list_kwargs)
        words_right_embeddings = self.get_embedding_list(words_right, **embd_list_kwargs)

        layerwise_scores = []
        num_layers = len(words_left_embeddings)
        all_layer_scores = [] #ADDED BY STEVEN

        with open(self.args.output_file, 'a') as file:
            file.write(f'Eval Dataset: {which_ds}\n\n')

            for layer in range(num_layers):
                if self.args.sim_metric == 'cosine':
                    cosine_similarities = [
                            cosine_similarity(
                                words_left_embeddings[layer][i],
                                words_right_embeddings[layer][i])\
                            for i in range(len(words_right_embeddings[layer]))]
                elif self.args.sim_metric == 'l2':
                    cosine_similarities = [
                            -np.linalg.norm(
                                words_left_embeddings[layer][i]\
                                - words_right_embeddings[layer][i])\
                            for i in range(len(words_right_embeddings[layer]))]
                elif self.args.sim_metric == 'norm_l2':
                    def _l2_normalize(a):
                        return a / np.linalg.norm(a)
                    cosine_similarities = [
                            -np.linalg.norm(
                                _l2_normalize(words_left_embeddings[layer][i])\
                                - _l2_normalize(words_right_embeddings[layer][i]))\
                            for i in range(len(words_right_embeddings[layer]))]
                else:
                    raise NotImplementedError
                evaluation_score = spearmanr(
                        cosine_similarities,
                        relatedness)
                #print(which_ds, add_top_blank, evaluation_score[0]) #COMMENTED OUT BY STEVEN
                file.write(f"{which_ds}, {add_top_blank}, {evaluation_score[0]}\n")
                all_layer_scores.append(evaluation_score[0]) #ADDED BY STEVEN
                layerwise_scores.append(
                        (evaluation_score[0], cosine_similarities))
            max_layer_score = max(all_layer_scores)
            print(f"max layer score for {which_ds}: {max_layer_score}") #ADDED BY STEVEN
            file.write(f"\nmax layer score for {which_ds}: {max_layer_score}\n\n\n")
            append_row_to_csv(self.args.output_csv, self.args.ckpt_path, which_ds, max_layer_score)
        
        return layerwise_scores, max_layer_score

    def do_one_eval(self, results):
        best_layer_score_lst = []
        for which_ds in self.string_lists_sims:
            if which_ds in results:
                continue
            res_w_b, max_layer_score = self.get_layerwise_scores_for_one_model_one_ds(
                    which_ds, add_top_blank=True)
            results[which_ds] = dict(
                    res_w_b=res_w_b,
                    )
            best_layer_score_lst.append(max_layer_score)
        final_model_avg_score = np.average(best_layer_score_lst)
        append_row_to_final_avg_csv(self.args.output_final_avg_score_csv, self.args.ckpt_path, final_model_avg_score)
        return results

    def eval_all(self):
        args = self.args

        self.get_result_path()
        result_path = self.result_path

        if not args.all_ckpts:
            results = {}
            if (not args.overwrite) and (os.path.exists(result_path)):
                results = pickle.load(open(result_path, 'rb'))
            results = self.do_one_eval(results)
            if result_path is not None:
                os.system('mkdir -p ' + os.path.dirname(result_path))
                pickle.dump(results, open(result_path, 'wb'))
        else:
            now_results = {}
            os.system('mkdir -p ' + os.path.dirname(result_path))
            if (not args.overwrite) and (os.path.exists(result_path)):
                now_results = pickle.load(open(result_path, 'rb'))

            for _ckpt in tqdm(self.all_ckpts):
                if _ckpt not in now_results:
                    now_results[_ckpt] = {}
                self.curr_ckpt = _ckpt
                self.lm.load_ckpt(os.path.join(self.exp_folder, _ckpt))
                now_results[_ckpt] = self.do_one_eval(now_results[_ckpt])
                pickle.dump(now_results, open(result_path, 'wb'))
        print(f'Results saved to {result_path}')


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.output_file):
        with open(args.output_file, 'w') as file:
            file.write('')
        print(f"File '{args.output_file}' created.")
    else:
        print(f"File '{args.output_file}' already exists.")

    with open(args.output_file, 'a') as file:
        file.write(f'\n\n\n\nWord Relatedness Scores for: {args.ckpt_path}\n\n')
    check_and_create_csv(args.output_csv)
    check_and_create_final_avg_csv(args.output_final_avg_score_csv)
    
    runner = WordSimRunner(args)
    runner.eval_all()

if __name__ == '__main__':
    main()
