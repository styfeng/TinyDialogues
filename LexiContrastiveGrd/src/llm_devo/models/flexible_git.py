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
import functools
import numpy as np
import transformers.models.git.modeling_git as modeling_git
try:
    from transformers.models.git.modeling_git import _expand_mask
except:
    # later versions of transformers
    from transformers.models.git.modeling_git import _prepare_4d_attention_mask as _expand_mask
import transformers.models.vit.modeling_vit as modeling_vit
import torch.nn.functional as nn_F

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
from ..env_vars import DEBUG


class GitModel(modeling_git.GitModel):
    def __init__(self, config):
        modeling_git.GitPreTrainedModel.__init__(
                self, config)
        self.config = config

        self.embeddings = modeling_git.GitEmbeddings(config)
        self.image_encoder = modeling_git.GitVisionModel(config.vision_config)
        self.encoder = modeling_git.GitEncoder(config)

        vis_proj_cls = self.get_visual_projector_cls()
        self.visual_projection = vis_proj_cls(config)

        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
                for _ in range(config.num_image_with_embedding)
            )
        self.visual_feature_method = getattr(
                config, 'visual_feature_method',
                'from_last_hidden_state')
        if self.visual_feature_method == 'from_cached_hidden_states':
            self.load_cached_vis_states()

        # Initialize weights and apply final processing
        self.post_init()

    def load_cached_vis_states(self):
        cached_hidden_states_path = self.config.cached_visual_states_path
        if cached_hidden_states_path is None:
            return
        tmp_contents = pickle.load(
                open(cached_hidden_states_path, 'rb'))
        self.cached_vis_hidden_states = torch.from_numpy(
                np.asarray(tmp_contents['states']))
        if torch.cuda.is_available():
            self.cached_vis_hidden_states = self.cached_vis_hidden_states.cuda()
        if self.cached_vis_hidden_states.ndim == 2:
            self.cached_vis_hidden_states\
                    = self.cached_vis_hidden_states.unsqueeze(dim=1)
        self.valid_idxs_for_vis_states = tmp_contents['valid_idxs']
        self.valid_idxs_for_vis_states_map = {
                valid_idx: _idx\
                for _idx, valid_idx in enumerate(tmp_contents['valid_idxs'])}

        self.cached_special_vis_hidden_states = {}
        if getattr(self.config, 'cached_special_visual_states_path', None) is None:
            return
        for sp_key, sp_path in self.config.cached_special_visual_states_path.items():
            loaded_states = pickle.load(open(sp_path, 'rb'))
            if isinstance(sp_key, int):
                loaded_states = torch.from_numpy(loaded_states)
                if torch.cuda.is_available():
                    loaded_states = loaded_states.cuda()
                if loaded_states.ndim == 1:
                    loaded_states = loaded_states.unsqueeze(dim=0)
                self.cached_special_vis_hidden_states[sp_key] = loaded_states
            elif isinstance(sp_key, str):
                offset = -int(sp_key.split('_')[-1])
                for _idx in range(len(loaded_states['words'])):
                    now_embd = loaded_states['embds'][_idx]
                    now_embd = torch.from_numpy(now_embd)
                    if torch.cuda.is_available():
                        now_embd = now_embd.cuda()
                    if now_embd.ndim == 3:
                        now_embd = now_embd.squeeze(dim=0)
                    self.cached_special_vis_hidden_states[offset - _idx] = now_embd
            else:
                raise NotImplementedError

    def get_visual_projector_cls(self):
        cls_name = getattr(
                self.config,
                'visual_projector_cls', None)
        if cls_name is None:
            cls = modeling_git.GitProjection
        else:
            cls = get_cls_by_name(cls_name)
        return cls

    def get_visual_features(self, pixel_values):
        projected_visual_features = None
        if pixel_values is not None:
            if pixel_values.ndim == 4:
                # here we assume pixel_values is of shape (batch_size, num_channels, height, width)
                if self.visual_feature_method == 'from_last_hidden_state':
                    visual_features = self.image_encoder(pixel_values).last_hidden_state
                elif self.visual_feature_method == 'from_all_hidden_states':
                    visual_features = self.image_encoder(
                            pixel_values,
                            output_hidden_states=True).hidden_states
                else:
                    raise NotImplementedError

            elif pixel_values.ndim == 5:
                raise NotImplementedError
                # here we assume pixel_values is of shape (batch_size, num_frames, num_channels, height, width)
                visual_features = []
                for frame_idx in range(pixel_values.shape[1]):
                    visual_features_frame = self.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
                    visual_features_frame += self.img_temperal_embedding[frame_idx]
                    visual_features.append(visual_features_frame)

                # finally, concatenate all features along sequence dimension
                visual_features = torch.cat(visual_features, dim=1)

            else:
                raise ValueError("pixel_values must be of rank 4 or 5")

            projected_visual_features = self.visual_projection(visual_features)
        return projected_visual_features

    def get_cached_visual_features_wo_proj(self, valid_idxs):
        valid_idxs = valid_idxs.cpu().numpy()
        visual_features = []
        for _idx in valid_idxs:
            if _idx not in self.cached_special_vis_hidden_states:
                new_idx = self.valid_idxs_for_vis_states_map[_idx]
                now_feat = self.cached_vis_hidden_states[new_idx]
            else:
                now_feat = self.cached_special_vis_hidden_states[_idx]
            visual_features.append(now_feat)
        visual_features = torch.stack(visual_features, 0)
        visual_features = visual_features.detach()
        return visual_features

    def get_visual_features_from_cached(self, valid_idxs):
        visual_features = GitModel.get_cached_visual_features_wo_proj(
                self, valid_idxs)
        projected_visual_features = self.visual_projection(visual_features)
        return projected_visual_features

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        valid_idxs: Optional[torch.Tensor] = None, 
        visual_embds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import requests
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")
        >>> model = AutoModel.from_pretrained("microsoft/git-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = "this is an image of two cats"

        >>> inputs = processor(text, images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if (valid_idxs is None) and (visual_embds is None):
            projected_visual_features = self.get_visual_features(pixel_values)
        elif (valid_idxs is not None):
            projected_visual_features = self.get_visual_features_from_cached(valid_idxs)
        else:
            projected_visual_features = self.visual_projection(visual_embds)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if projected_visual_features is None:
            projected_visual_features = torch.zeros(
                (embedding_output.shape[0], 0, embedding_output.shape[2]),
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )

        # Repeat visual features to match embedding batch size.
        projected_visual_features = projected_visual_features.repeat(
            embedding_output.size(0) // projected_visual_features.size(0), 1, 1
        )

        # concatenate patch token and text token embeddings
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)

        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)

        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_visual_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                    attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(
                embedding_output.device
            )
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :] += expanded_attn_mask

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=combined_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values_present=pixel_values is not None,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GitForCausalLM(modeling_git.GitForCausalLM):
    def __init__(self, config):
        modeling_git.GitPreTrainedModel.__init__(self, config)

        self.git = GitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

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
            valid_idxs: Optional[torch.Tensor] = None, 
            visual_embds: Optional[torch.Tensor] = None, 
            ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
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
                valid_idxs=valid_idxs,
                visual_embds=visual_embds,
                )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            if (pixel_values is not None) or (valid_idxs is not None) or (visual_embds is not None):
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NoImgGitForCausalLM(GitForCausalLM):
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
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        simple_forward = getattr(
                self.config, 'use_simple_forward',
                False)
        if not simple_forward:
            add_fwd_kwargs = dict(
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    )
        else:
            add_fwd_kwargs = {}
        outputs = self.git(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **add_fwd_kwargs)

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens\
                               if pixel_values is not None else 0
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            #past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlexVisualProjector(nn.Module):
    def __init__(
            self, config,
            ):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps),
        )
        self.with_bottleneck = getattr(self.config, 'flex_vis_proj_bn', None)
        self.num_hidden_feats = getattr(
                self.config, 
                'flex_vis_proj_hid_feats',
                None)
        input_shape = getattr(
                self.config,
                'flex_vis_proj_input_shape', 197)
        if self.with_bottleneck is not None:
            self.bottle_neck = nn.Parameter(
                    torch.rand(
                        input_shape * self.num_hidden_feats,
                        self.with_bottleneck))

    def forward(self, embeddings) -> torch.Tensor:
        if self.with_bottleneck is not None:
            embeddings = embeddings[-self.num_hidden_feats:]
            embeddings = torch.cat(embeddings, dim=1)
            embeddings = torch.transpose(embeddings, 1, 2)
            embeddings = torch.matmul(embeddings, self.bottle_neck)
            embeddings = torch.transpose(embeddings, 1, 2)
        return self.visual_projection(embeddings)


class NonLinearPooler(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
                 for _ in range(num_layers)])
        self.layers.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=0.02,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class GitEmbeddingsNoPos(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GitEmbeddingsCopy(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GitTextModel(modeling_git.GitModel):
    def __init__(self, config):
        modeling_git.GitPreTrainedModel.__init__(
                self, config)
        self.config = config

        embd_type = getattr(config, 'pos_embed_type', None)
        if embd_type is None:
            self.embeddings = modeling_git.GitEmbeddings(config)
        elif embd_type == 'relative':
            config.position_embedding_type = 'relative_key_query'
            self.embeddings = modeling_git.GitEmbeddings(config)
        else:
            raise NotImplementedError
        self.encoder = modeling_git.GitEncoder(config)
        self.att_wd_size = getattr(config, 'att_wd_size', None)
        self.post_init()

    def get_combined_attention_mask(
            self, seq_length, embedding_output,
            input_shape, attention_mask, past_key_values_length):
        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)

        projected_visual_features = torch.zeros(
            (embedding_output.shape[0], 0, embedding_output.shape[2]),
            dtype=embedding_output.dtype,
            device=embedding_output.device,
        )
        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_visual_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                    attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(
                embedding_output.device
            )
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :] += expanded_attn_mask
        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        hidden_states = embedding_output

        combined_attention_mask = self.get_combined_attention_mask(
            seq_length, embedding_output,
            input_shape, attention_mask, past_key_values_length)

        if self.att_wd_size is not None:
            combined_attention_mask += _make_att_wd_mask(
                    input_shape, hidden_states.dtype,
                    device=hidden_states.device,
                    att_wd_size=self.att_wd_size)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=combined_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values_present=None,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        last_hidden_state = sequence_output
        eos_token_id = getattr(self.config, 'eos_token_id', 102) # default eos token id for git tokenizer
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VLMClassificationHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn_F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class VokenGitForCausalLM(modeling_git.GitForCausalLM):
    def __init__(self, config):
        modeling_git.GitPreTrainedModel.__init__(self, config)

        self.git = GitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        voken_size = getattr(config, 'voken_size', 50000)
        config.voken_size = voken_size
        self.voken_output = VLMClassificationHead(config)
        self.loss_fct = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

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
            valid_idxs: Optional[torch.Tensor] = None, 
            visual_embds: Optional[torch.Tensor] = None, 
            vokens: Optional[torch.Tensor] = None,
            ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
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
                output_hidden_states=True,
                return_dict=return_dict,
                valid_idxs=valid_idxs,
                visual_embds=visual_embds,
                )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)
        if vokens is not None:
            voken_readout_layer = getattr(
                    self.config, 'voken_readout_layer',
                    None)
            if voken_readout_layer is None:
                voken_logits = self.voken_output(
                        sequence_output)
            elif isinstance(voken_readout_layer, int):
                voken_logits = self.voken_output(
                        outputs.hidden_states[voken_readout_layer])

        loss = None
        if labels is not None:
            num_image_tokens = 0
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = self.loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if vokens is not None:
            voken_predict_method = getattr(self.config, 'voken_predict_method', None)
            if voken_predict_method is None:
                shifted_logits = voken_logits[:, :-1, :].contiguous()
                labels = vokens[:, 1:].contiguous()
            elif voken_predict_method == 'no_shift':
                shifted_logits = voken_logits.contiguous()
                labels = vokens.contiguous()
            else:
                raise NotImplementedError
            vk_loss = self.loss_fct(shifted_logits.view(-1, self.config.voken_size), labels.view(-1))
            if loss is None:
                loss = vk_loss
            else:
                loss = loss + vk_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
