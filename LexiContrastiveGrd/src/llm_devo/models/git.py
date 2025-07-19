import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from typing import List, Optional, Tuple, Union
import warnings
import ipdb
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from itertools import product
import numpy as np
import transformers.models.git.modeling_git as modeling_git
import transformers.models.vit.modeling_vit as modeling_vit
from transformers.models.opt.modeling_opt import OPTConfig
import transformers.models.opt.modeling_opt as hg_opt
import transformers.models.clip.modeling_clip as modeling_clip
from .utils import freeze_model_and_make_eval_
from . import flexible_git


class GitForCausalLM(modeling_git.GitForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_git.CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.git(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            if pixel_values is not None:
                num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
            else:
                num_image_tokens = 0
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return modeling_git.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_noimg_more_lyrs_git_func(
        git_model_size='large',
        num_layers=12,
        output_with_bias=True,
        tie_output=False,
        ):
    model_name = f"microsoft/git-{git_model_size}"
    config = modeling_git.GitConfig.from_pretrained(model_name)
    config.num_hidden_layers = num_layers
    config.vocab_size = 32778
    model = GitForCausalLM(config)
    del model.git.image_encoder
    del model.git.visual_projection
    if not output_with_bias:
        if tie_output:
            config.tie_word_embeddings = True
        del model.output
        model.output = nn.Linear(
                config.hidden_size, config.vocab_size,
                bias=False)
        model.post_init()
    model.train()
    model.git.encoder.layer[0].attention.self.image_patch_tokens = 0
    return model


def change_to_tie_output(model):
    model.config.tie_word_embeddings = True
    del model.output
    model.output = nn.Linear(
            model.config.hidden_size,
            model.config.vocab_size,
            bias=False)
    model.post_init()
    return model


def change_to_dino(
        model, dino_model_name,
        reinit_vis_proj=True):
    del model.git.image_encoder
    model.git.image_encoder = ViTModel.from_pretrained(dino_model_name)
    dino_cfg = model.git.image_encoder.config
    config = model.git.config
    config.vision_config.hidden_size = dino_cfg.hidden_size
    if reinit_vis_proj:
        del model.git.visual_projection
        model.git.visual_projection = modeling_git.GitProjection(config)
        num_tks = (dino_cfg.image_size // dino_cfg.patch_size) ** 2 + 1
        model.git.encoder.layer[0].attention.self.image_patch_tokens = num_tks
    return model


def get_dino_git_func(
        git_model_size='large',
        dino_model_name='facebook/dino-vitb16',
        tie_output=False,
        num_layers=None,
        ):
    model_name = f"microsoft/git-{git_model_size}"
    config = modeling_git.GitConfig.from_pretrained(model_name)
    if num_layers is not None:
        config.num_hidden_layers = num_layers
    config.vocab_size = 32778
    model = GitForCausalLM(config)
    model = change_to_dino(model, dino_model_name)
    model.train()
    freeze_model_and_make_eval_(model.git.image_encoder)
    if tie_output:
        model = change_to_tie_output(model)
    return model


def get_dino_cached_git_func(
        git_model_size='large',
        hidden_states_fname='conceptual_12m.get_CC12M_15M_base.pkl',
        ave_cache_size='15M',
        cache_source='dino',
        ):
    model_name = f"microsoft/git-{git_model_size}"
    config = modeling_git.GitConfig.from_pretrained(model_name)
    config.visual_feature_method = 'from_cached_hidden_states'
    from llm_devo.env_vars import DATASET_ROOT_DIR_FREQ
    config.cached_visual_states_path = os.path.join(
            DATASET_ROOT_DIR_FREQ, 'Conceptual-12M',
            'cached_hidden_states',
            cache_source, hidden_states_fname)
    if cache_source in ['dino', 'dino_v2']:
        config.vision_config.hidden_size = 768
    elif cache_source == 'dino_res50':
        config.vision_config.hidden_size = 2048
    config.tie_word_embeddings = True
    model = flexible_git.GitForCausalLM(config)
    del model.git.image_encoder
    model.git.encoder.layer[0].attention.self.image_patch_tokens\
            = model.git.cached_vis_hidden_states.shape[1]
    del model.output
    model.output = nn.Linear(
            config.hidden_size, config.vocab_size,
            bias=False)
    model.post_init()
    model.train()
    return model
