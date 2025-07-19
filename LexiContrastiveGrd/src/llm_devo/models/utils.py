import torch
import importlib

def exists(val):
    return val is not None

# for controlling freezing during training of flamingo

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

def _make_att_wd_mask(
        input_ids_shape: torch.Size, 
        dtype: torch.dtype, device: torch.device, 
        past_key_values_length: int = 0,
        att_wd_size: int = 0,
    ):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(
            mask_cond > (mask_cond - att_wd_size).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def get_cls_by_name(cls_name):
    if callable(cls_name):
        return cls_name
    module_name = '.'.join(cls_name.split('.')[:-1])
    func_name = cls_name.split('.')[-1]
    load_setting_module = importlib.import_module(module_name)
    cls = getattr(load_setting_module, func_name)
    return cls
