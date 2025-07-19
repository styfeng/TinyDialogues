import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from typing import List, Optional, Tuple, Union
import warnings
import pickle
import ipdb
import os
import torch
from torch import nn
from itertools import product
import numpy as np
import transformers.models.clip.modeling_clip as modeling_clip
import transformers.models.git.modeling_git as modeling_git
import transformers.models.vit.modeling_vit as modeling_vit

from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.git.configuration_git import GitConfig, GitVisionConfig
from torch.nn import CrossEntropyLoss

from .utils import exists, set_module_requires_grad_, freeze_all_layers_,\
        unfreeze_all_layers_, freeze_model_and_make_eval_, _make_att_wd_mask,\
        get_cls_by_name
from .flexible_git import GitModel, NonLinearPooler, GitTextModel
from ..env_vars import get_text_eval, DEBUG


class FlexCLIPBase(modeling_clip.CLIPModel):
    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_text)
        return logits_per_text, logits_per_image, loss

    def text_output_to_embd(
            self, text_outputs):
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds


class CLIPFromVisCache(FlexCLIPBase):
    def __init__(self, config: modeling_clip.CLIPConfig):
        modeling_clip.CLIPPreTrainedModel.__init__(
                self, config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.config = config
        if not get_text_eval():
            GitModel.load_cached_vis_states(self)
        if getattr(config, 'change_txt_builder', None) is None:
            config.change_txt_builder = None
        if config.change_txt_builder is None:
            self.text_model = modeling_clip.CLIPTextTransformer(text_config)
        elif config.change_txt_builder == 'GITEncoder':
            self.text_model = GitTextModel(text_config)
            self.text_model.encoder.layer[0].attention.self.image_patch_tokens = 0
        elif config.change_txt_builder == 'GITEncoder_AttWd':
            text_config.att_wd_size = 2
            self.text_model = GitTextModel(text_config)
            self.text_model.encoder.layer[0].attention.self.image_patch_tokens = 0
        elif config.change_txt_builder == 'GITEncoder_AttWd4':
            text_config.att_wd_size = 4
            self.text_model = GitTextModel(text_config)
            self.text_model.encoder.layer[0].attention.self.image_patch_tokens = 0
        elif config.change_txt_builder == 'GITEncoder_AttWd6':
            text_config.att_wd_size = 6
            self.text_model = GitTextModel(text_config)
            self.text_model.encoder.layer[0].attention.self.image_patch_tokens = 0
        else:
            raise NotImplementedError

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()
        if getattr(config, 'nonlinear_pooler_layers', 0) > 0:
            nonlinear_pooler = NonLinearPooler(
                    hidden_dim=self.vision_embed_dim,
                    num_layers=config.nonlinear_pooler_layers)
            self.visual_projection = nn.Sequential(
                    nonlinear_pooler,
                    self.visual_projection)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        visual_embds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, modeling_clip.CLIPOutput]:
        self.tmp_input_ids = input_ids
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if visual_embds is None:
            image_embeds = GitModel.get_visual_features_from_cached(
                    self, valid_idxs).squeeze(dim=1)
        else:
            image_embeds = self.visual_projection(visual_embds).squeeze(dim=1)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = self.text_output_to_embd(text_outputs)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text, logits_per_image, loss\
                = self.get_logits_and_loss(
                    text_embeds, image_embeds,
                    return_loss)
        self.tmp_input_ids = None

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs)
            return ((loss,) + output) if loss is not None else output

        return modeling_clip.CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
        )

class CLIPSimpleFwd(FlexCLIPBase):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, modeling_clip.CLIPOutput]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if getattr(self.config, 'vis_use_last_hidden', None) is None:
            image_embeds = vision_outputs[1]
            image_embeds = self.visual_projection(image_embeds)
        else:
            image_embeds = vision_outputs.last_hidden_state
            image_embeds = self.visual_projection(image_embeds)
            if image_embeds.ndim == 3:
                image_embeds = image_embeds.squeeze(dim=1)

        text_embeds = self.text_output_to_embd(text_outputs)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text, logits_per_image, loss\
                = self.get_logits_and_loss(
                    text_embeds, image_embeds,
                    return_loss)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return modeling_clip.CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class AvgPooler(nn.Module):
    def __init__(
            self, hidden_size,
            spatial_avg=True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.spatial_avg = spatial_avg

    def forward(self, hidden_states):
        # average across the last 2 dims
        if self.spatial_avg:
            avg_tensor = torch.mean(hidden_states, dim=(2, 3))
        else:
            avg_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(avg_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SpKeepPooler(nn.Module):
    def __init__(
            self, hidden_size,
            ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = hidden_states.reshape(
                    hidden_states.shape[0],
                    -1,
                    hidden_states.shape[-1])
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class CLIPSpatialMax(CLIPSimpleFwd):
    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        logits_per_image, _ = torch.max(logits_per_image, dim=1)
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_text)
        return logits_per_text, logits_per_image, loss


class CLIPSpatialMean(CLIPSimpleFwd):
    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        logits_per_image = torch.mean(logits_per_image, dim=1)
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_text)
        return logits_per_text, logits_per_image, loss


class CLIPSpatialSoftMax(CLIPSimpleFwd):
    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        sftmx_logits_per_image = nn.functional.softmax(logits_per_image, dim=1)
        logits_per_image = torch.sum(
                sftmx_logits_per_image * logits_per_image,
                dim=1)
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_text)
        return logits_per_text, logits_per_image, loss


class CLIPTextMax(CLIPSimpleFwd):
    def text_output_to_embd(
            self, text_outputs):
        text_embeds = text_outputs[0]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds

    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_text, _ = torch.max(logits_per_text, dim=1)
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_text)
        return logits_per_text, logits_per_image, loss


class CLIPFromVisCacheTextSep(CLIPFromVisCache):
    def __init__(self, config):
        super().__init__(config)
        self.cache_loss_mask = {}

    def text_output_to_embd(
            self, text_outputs):
        text_embeds = text_outputs[0]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds

    def get_logits_select_mask(
            self, bs_d, seq_d,
            logits):
        bs_d, seq_d = int(bs_d), int(seq_d)
        cache_key = (bs_d, seq_d)
        if cache_key in self.cache_loss_mask:
            return self.cache_loss_mask[cache_key]
        overall_mask = torch.ones(
                bs_d * seq_d, bs_d * seq_d,
                device=logits.device)
        for idx in range(bs_d):
            for add_idx_x, add_idx_y in product(range(seq_d), range(seq_d)):
                if add_idx_x != add_idx_y:
                    overall_mask[\
                            idx * seq_d + add_idx_x,\
                            idx * seq_d + add_idx_y,\
                            ] = 0
        self.cache_loss_mask[cache_key] = overall_mask
        return overall_mask

    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        sep_loss_type = getattr(
                self.config,
                'sep_loss_type', 'simple_repeat')
        if sep_loss_type == 'simple_repeat':
            bs_d, seq_d, _ = logits_per_text.shape
            logits_per_text = logits_per_text.unsqueeze(-2).repeat(1, 1, seq_d, 1)
            logits_per_text = logits_per_text.reshape(
                    bs_d * seq_d, seq_d, bs_d)
            logits_per_text = logits_per_text.reshape(
                    bs_d * seq_d, seq_d * bs_d)
            logits_per_image = logits_per_text.t()
        elif sep_loss_type in ['pos_neg_crct', 'pos_neg_crct_ignore_pad']:
            bs_d, seq_d, _ = logits_per_text.shape
            logits_per_text_for_loss = logits_per_text.reshape(
                    bs_d * seq_d, bs_d)
            label_for_text = torch.arange(
                    bs_d, device=logits_per_text_for_loss.device)
            label_for_text = label_for_text.unsqueeze(1).repeat(1, seq_d)
            label_for_text = label_for_text.reshape(seq_d * bs_d)
            if sep_loss_type == 'pos_neg_crct_ignore_pad':
                ids_reshape = self.tmp_input_ids.reshape(bs_d * seq_d)
                ignore_mask = ids_reshape == self.config.tokenizer_pad_token_id
                label_for_text[ignore_mask] = -100

            sample_loss_num = getattr(
                    self.config, 'sample_loss_num',
                    None)
            if sample_loss_num is None:
                loss_text = nn.functional.cross_entropy(
                        logits_per_text_for_loss,
                        label_for_text)
            else:
                sample_idx = torch.randperm(bs_d * seq_d)[:sample_loss_num]
                loss_text = nn.functional.cross_entropy(
                        logits_per_text_for_loss[sample_idx],
                        label_for_text[sample_idx])

            logits_per_text = logits_per_text.unsqueeze(-2).repeat(1, 1, seq_d, 1)
            logits_per_text = logits_per_text.reshape(
                    bs_d * seq_d, seq_d, bs_d)
            logits_per_text = logits_per_text.reshape(
                    bs_d * seq_d, seq_d * bs_d)
            logits_per_image = logits_per_text.t()

            select_mask = self.get_logits_select_mask(bs_d, seq_d, logits_per_image)
            logits_per_image = logits_per_image[select_mask.to(torch.bool)]
            logits_per_image = logits_per_image.reshape(
                    bs_d * seq_d, (bs_d - 1)*seq_d + 1)
            label_for_image = label_for_text * seq_d
            if sep_loss_type == 'pos_neg_crct_ignore_pad':
                label_for_image[ignore_mask] = -100
            if sample_loss_num is None:
                loss_image = nn.functional.cross_entropy(
                        logits_per_image,
                        label_for_image)
            else:
                loss_image = nn.functional.cross_entropy(
                        logits_per_image[sample_idx],
                        label_for_image[sample_idx])

        loss = None
        if return_loss:
            if sep_loss_type == 'simple_repeat':
                loss = modeling_clip.clip_loss(logits_per_text)
            elif sep_loss_type in ['pos_neg_crct', 'pos_neg_crct_ignore_pad']:
                loss = loss_text + loss_image
        #print(loss_text, loss_image)
        #print(torch.max(label_for_image[sample_idx]), logits_per_image.shape)
        return logits_per_text, logits_per_image, loss


class CLIPFromVisCacheTextSepWCausal(CLIPFromVisCacheTextSep):
    def __init__(self, config: modeling_clip.CLIPConfig):
        self.output = None
        super().__init__(config)
        self.setup_causal_decoder()

    def get_decoder_class(self):
        which_decoder_class = getattr(
                self.config, 'causal_decoder_class',
                'CLIPEncoder')
        return which_decoder_class

    def setup_causal_decoder(self):
        causal_config = self.config.causal_text_config
        which_decoder_class = self.get_decoder_class()
        if which_decoder_class == 'CLIPEncoder':
            self.causal_text_decoder = modeling_clip.CLIPEncoder(causal_config)
        elif which_decoder_class == 'GitEncoder':
            self.causal_text_decoder = modeling_git.GitEncoder(causal_config)
            self.causal_text_decoder.layer[0].attention.self.image_patch_tokens = 0
        else:
            raise NotImplementedError

        self.output = nn.Linear(
                causal_config.hidden_size, causal_config.vocab_size,
                bias=False)
        self.config.tie_word_embeddings = True
        self.post_init()

    def get_input_embeddings(self):
        txt_builder = self.config.change_txt_builder
        if (txt_builder is None) or ('GITEncoder' not in txt_builder):
            return self.text_model.embeddings.token_embedding
        else:
            return self.text_model.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.output

    def get_text_outputs(
            self, 
            input_ids,
            attention_mask,
            position_ids,
            output_attentions,
            output_hidden_states,
            return_dict):
        text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        return text_outputs

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def create_attention_mask(self, *args, **kwargs):
        return GitTextModel.create_attention_mask(
                self, *args, **kwargs)

    def get_text_causal_outputs(
            self, text_outputs, attention_mask):
        hidden_states = text_outputs[0]
        which_decoder_class = self.get_decoder_class()
        if which_decoder_class == 'CLIPEncoder':
            causal_attention_mask = modeling_clip._make_causal_mask(
                    input_ids.size(), hidden_states.dtype, device=hidden_states.device)
            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = modeling_clip._expand_mask(attention_mask, hidden_states.dtype)
            text_causal_output = self.causal_text_decoder(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask)
        elif which_decoder_class == 'GitEncoder':
            combined_attention_mask = GitTextModel.get_combined_attention_mask(
                self, hidden_states.shape[1], hidden_states,
                hidden_states.shape[:2],
                attention_mask, past_key_values_length=0)
            text_causal_output = self.causal_text_decoder(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    pixel_values_present=None,
                    output_hidden_states=True,
                    )
        else:
            raise NotImplementedError
        return text_causal_output

    def get_causal_loss(
            self, 
            causal_logits,
            labels):
        shifted_logits = causal_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        causal_loss = loss_fct(
                shifted_logits.view(-1, self.config.text_config.vocab_size),
                labels.view(-1))
        return causal_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        visual_embds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, modeling_clip.CLIPOutput]:
        self.tmp_input_ids = input_ids

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (valid_idxs is not None) or (visual_embds is not None):
            if visual_embds is None:
                image_embeds = GitModel.get_visual_features_from_cached(
                        self, valid_idxs).squeeze(dim=1)
            else:
                image_embeds = self.visual_projection(visual_embds).squeeze(dim=1)
            have_img = True
        else:
            have_img = False

        text_outputs = self.get_text_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_causal_output = self.get_text_causal_outputs(
                text_outputs, attention_mask)
        causal_seq_output = text_causal_output[0]
        causal_logits = self.output(causal_seq_output)

        if labels is None:
            labels = input_ids
        causal_loss = self.get_causal_loss(
                causal_logits,
                labels=labels)

        if have_img:
            text_embeds = self.text_output_to_embd(text_outputs)
            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_text, logits_per_image, loss\
                    = self.get_logits_and_loss(
                        text_embeds, image_embeds,
                        return_loss)
        else:
            image_embeds = None
            text_embeds = None
            logits_per_text, logits_per_image = None, None
            loss = 0
        self.tmp_input_ids = None

        clip_loss_weight = getattr(
                self.config, 'clip_loss_weight', 1.0)
        causal_loss_weight = getattr(
                self.config, 'causal_loss_weight', 1.0)
        if return_loss:
            loss = clip_loss_weight * loss + causal_loss_weight * causal_loss

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs)
            return ((loss,) + output) if loss is not None else output

        return modeling_clip.CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
        )


class CLIPFromVisCacheOnlyCausal(CLIPFromVisCacheTextSepWCausal):
    def __init__(self, config: modeling_clip.CLIPConfig):
        self.output = None
        super().__init__(config)
        del self.visual_projection
        del self.text_projection
        del self.logit_scale

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        visual_embds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, modeling_clip.CLIPOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        text_outputs = self.get_text_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_causal_output = self.get_text_causal_outputs(
                text_outputs, attention_mask)
        causal_seq_output = text_causal_output[0]
        causal_logits = self.output(causal_seq_output)

        causal_loss = self.get_causal_loss(
                causal_logits,
                labels=input_ids)
        if return_loss:
            loss = causal_loss

        if not return_dict:
            output = (text_outputs,)
            return ((loss,) + output) if loss is not None else output

        return modeling_clip.CLIPOutput(
            loss=loss,
            text_model_output=text_outputs,
        )


class CLIPFromVisCacheWCausal(CLIPFromVisCacheTextSepWCausal):
    def __init__(self, config: modeling_clip.CLIPConfig):
        self.output = None
        super().__init__(config)

    def get_logits_and_loss(
            self,
            text_embeds, image_embeds,
            return_loss):
        return FlexCLIPBase.get_logits_and_loss(
                self,
                text_embeds, image_embeds,
                return_loss)

    def text_output_to_embd(
            self, text_outputs):
        return FlexCLIPBase.text_output_to_embd(self, text_outputs)


class CLIPTextSepWCausal(CLIPFromVisCacheTextSepWCausal):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        visual_embds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        valid_idxs: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.get_text_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_causal_output = self.get_text_causal_outputs(
                text_outputs, attention_mask)
        if text_outputs.hidden_states is not None:
            all_hidden_states = tuple(
                    list(text_outputs.hidden_states)\
                    + list(text_causal_output.hidden_states))
        else:
            all_hidden_states = None

        causal_seq_output = text_causal_output[0]
        causal_logits = self.output(causal_seq_output)
        if labels is None:
            causal_loss = None
        else:
            causal_loss = self.get_causal_loss(
                    causal_logits,
                    labels=labels)

        if not return_dict:
            output = (causal_logits,) + outputs[1:]
            return ((causal_loss,) + output) if causal_loss is not None else output

        return CausalLMOutputWithPast(
            loss=causal_loss,
            logits=causal_logits,
            hidden_states=all_hidden_states,
            attentions=text_outputs.attentions,
        )


class CLIPTextEval(CLIPFromVisCache):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        valid_idxs: Optional[torch.Tensor] = None,
        visual_embds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
