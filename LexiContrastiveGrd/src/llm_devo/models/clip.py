import transformers
import copy
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
from transformers import ViTFeatureExtractor, ViTModel
from transformers.models.clip.modeling_clip import CLIPConfig
from transformers.models.git.configuration_git import GitConfig
import warnings
import ipdb
import os
import torch
from torch import nn
from itertools import product
import numpy as np
from .utils import exists, set_module_requires_grad_, freeze_all_layers_,\
        unfreeze_all_layers_, freeze_model_and_make_eval_
from ..env_vars import get_text_eval, DEBUG
from . import flexible_clip


def reset_clip_text_max_length(model, new_len=128):
    model.text_model.config.max_position_embeddings = new_len
    if getattr(
            model.text_model.embeddings,
            'position_embedding',
            None) is None:
        return model
    del model.text_model.embeddings.position_embedding
    del model.text_model.embeddings.position_ids
    model.text_model.embeddings.position_embedding = nn.Embedding(
            model.text_model.config.max_position_embeddings, 
            model.text_model.config.hidden_size)
    model.text_model.embeddings.register_buffer(
            "position_ids", 
            torch.arange(
                model.text_model.config.max_position_embeddings,
                ).expand((1, -1)))
    return model

def get_text_eval_clip(model_name, text_config=None):
    # Get CLIP model for text evaluation
    if text_config is None:
        model = CLIPTextModel.from_pretrained(model_name)
        model = reset_clip_text_max_length(model, new_len=128)
        model.text_model.apply(model._init_weights)
    else:
        model = CLIPTextModel(text_config)
        model = reset_clip_text_max_length(model, new_len=128)
    model.train()
    return model


def change_text_model_to_git_like(config):
    config.text_config.num_hidden_layers = 6
    config.text_config.hidden_size = 768
    config.text_config.intermediate_size = 3072
    config.text_config.num_attention_heads = 12
    config.text_config.vocab_size = 30522
    return config

def change_to_git_enc_w_causal_sp2k(config):
    config.causal_text_config = GitConfig.from_pretrained(
            "microsoft/git-large")
    config.causal_text_config.num_hidden_layers = 4
    config.causal_decoder_class = 'GitEncoder'
    config.sep_loss_type = 'pos_neg_crct_ignore_pad'
    config.tokenizer_pad_token_id = 0
    config.sample_loss_num = 2048
    return config

def get_cached_clip_git_like_func(
        clip_model_name='base-patch16',
        hidden_states_fname='conceptual_12m.get_CC12M_50M_base.pkl',
        nonlinear_pooler_layers=1,
        cache_source='dino',
        text_hid_layers=None,
        change_eos_token_id=None,
        text_max=False,
        change_txt_builder=None,
        hidden_size=None,
        ):
    model_name = f"openai/clip-vit-{clip_model_name}"
    config = CLIPConfig.from_pretrained(model_name)
    config = change_text_model_to_git_like(config)

    config.change_txt_builder = change_txt_builder
    if (change_txt_builder is not None)\
            and ('GITEncoder' in change_txt_builder):
        config.text_config = GitConfig.from_pretrained(
                "microsoft/git-large")
    if text_hid_layers is not None:
        config.text_config.num_hidden_layers = text_hid_layers
    if change_eos_token_id:
        config.text_config.eos_token_id = change_eos_token_id
    if get_text_eval():
        if ((text_max is False)\
            or (('sep_crt' not in text_max)\
                and ('_w_causal' not in text_max)))\
           and (change_txt_builder is None):
            return get_text_eval_clip(
                    model_name,
                    text_config=config.text_config)
    from llm_devo.env_vars import DATASET_ROOT_DIR_FREQ
    config.cached_visual_states_path = os.path.join(
            DATASET_ROOT_DIR_FREQ, 'Conceptual-12M',
            'cached_hidden_states',
            cache_source, hidden_states_fname)
    if cache_source == 'dino_res50':
        config.vision_config.hidden_size = 2048
    config.nonlinear_pooler_layers = nonlinear_pooler_layers
    
    if text_max is False:
        model_builder = flexible_clip.CLIPFromVisCache
    elif text_max == 'sep_crt_ip_sp2k_w_causal_d3v1_git_ly5':
        model_builder = flexible_clip.CLIPFromVisCacheTextSepWCausal
        config = change_to_git_enc_w_causal_sp2k(config)
        config.clip_loss_weight = 0.3
        config.causal_text_config.num_hidden_layers = 5
    else:
        raise NotImplementedError

    if DEBUG:
        print(get_text_eval())
        ipdb.set_trace()
    if get_text_eval():
        if (text_max is False) or ('_w_causal' not in text_max):
            model_builder = flexible_clip.CLIPTextEval
        else:
            if 'wgit' not in text_max:
                model_builder = flexible_clip.CLIPTextSepWCausal

    if hidden_size is not None:
        assert hidden_size % config.text_config.num_attention_heads == 0
        config.text_config.intermediate_size = hidden_size * 4
        config.text_config.hidden_size = hidden_size
        if getattr(config, 'causal_text_config', None) is not None:
            assert hidden_size % config.causal_text_config.num_attention_heads == 0
            config.causal_text_config.intermediate_size = hidden_size * 4
            config.causal_text_config.hidden_size = hidden_size
    
    model = model_builder(config)
    model = reset_clip_text_max_length(model, new_len=128)
    model.train()
    return model
