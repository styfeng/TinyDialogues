import os
import pdb
import setuptools
import torch

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from transformers import PreTrainedTokenizerFast
import llm_devo


def get_gpt2_tokenizer_func(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer


def get_roberta_tokenizer_func(model_name="roberta-base"):
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False
    tokenizer.add_special_tokens(
            {
                'bos_token': '<s>', 
                'unk_token': '<unk>',
                'additional_special_tokens': [
                    '<image>', '</c>', 
                    '<PERSON>', # C-12M for person names
                    ]
            })
    return tokenizer


def tk_pad_collate_fn(all_data, tokenizer):
    keys = list(all_data[0].keys())
    assert 'text' in keys
    all_texts = [_data['text'] for _data in all_data]
    all_texts = tokenizer(
            all_texts, padding='longest', return_tensors="pt", 
            truncation=True, max_length=128)
    all_texts = all_texts.input_ids
    ret_dict = dict(
            input_ids=all_texts, 
            labels=all_texts,
            )
    keys.remove('text')
    for other_key in keys:
        all_other_value = [_data[other_key] for _data in all_data]
        all_other_value = torch.stack(all_other_value, 0)
        ret_dict[other_key] = all_other_value
    return ret_dict

def get_git_tokenizer_func():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/git-large")
    return tokenizer

def get_git_w_img_tokenizer_func():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/git-large")
    tokenizer.add_special_tokens(
            {
                'additional_special_tokens': ['<image>'],
            })
    return tokenizer
