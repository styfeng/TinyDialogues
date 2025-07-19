import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch import einsum, nn
import ipdb


class MultiCombine(nn.Module):
    def __init__(
            self, 
            base_model_func,
            mix_weights=[1, 1],
            add_vis=True,
            ):
        super().__init__()
        self.base_model = base_model_func()
        self.config = self.base_model.config
        self.mix_weights = mix_weights
        self.add_vis = add_vis

    def save_pretrained(self, *args, **kwargs):
        self.base_model.save_pretrained(*args, **kwargs)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            valid_idxs: Optional[torch.Tensor] = None, 
            visual_embds: Optional[torch.Tensor] = None, 
            vokens: Optional[torch.Tensor] = None,
            noimg_input_ids: Optional[torch.Tensor] = None,
            noimg_attention_mask: Optional[torch.Tensor] = None,
            noimg_labels: Optional[torch.Tensor] = None,
            noimg_vokens: Optional[torch.Tensor] = None,
            noimg_token_type_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ):
        shared_fwd_kwargs = dict(
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        first_fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                )
        if self.add_vis:
            if self.add_vis == True:
                first_fwd_kwargs.update(dict(
                        pixel_values=pixel_values,
                        valid_idxs=valid_idxs,
                        visual_embds=visual_embds,
                        ))
            elif self.add_vis == 'only_pixel':
                first_fwd_kwargs.update(dict(
                        pixel_values=pixel_values,
                        ))
            elif self.add_vis == 'vokens':
                first_fwd_kwargs.update(dict(
                        vokens=vokens,
                        ))
            else:
                raise NotImplementedError
        first_fwd_kwargs.update(shared_fwd_kwargs)
        outputs = self.base_model(
                **first_fwd_kwargs)
        if outputs.loss is not None:
            now_loss = outputs.loss * self.mix_weights[0]
        else:
            now_loss = None

        if noimg_input_ids is not None:
            second_fwd_kwargs = dict(
                    input_ids=noimg_input_ids,
                    attention_mask=noimg_attention_mask,
                    labels=noimg_labels,
                    )
            second_fwd_kwargs.update(shared_fwd_kwargs)
            if self.add_vis == 'vokens':
                second_fwd_kwargs.update(dict(vokens=noimg_vokens))
            noimg_outputs = self.base_model(
                    **second_fwd_kwargs)
            if noimg_outputs.loss is not None:
                noimg_loss = noimg_outputs.loss * self.mix_weights[1]
                now_loss = now_loss + noimg_loss
        final_outputs = outputs # not including pure text outputs for now
        final_outputs.loss = now_loss
        return final_outputs


def get_multi_comb_model(*args, **kwargs):
    return MultiCombine(*args, **kwargs)
