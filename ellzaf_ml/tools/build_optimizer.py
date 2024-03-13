import json
from functools import partial
from torch import optim as optim

def build_pretrain_optimizer(model, lr):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    
    optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999), lr=lr, weight_decay=0.05)
    return optimizer


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    print(f'No decay params: {no_decay_name}')
    print(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def build_finetune_optimizer(model,lr, distil=False, vit=True):
    print('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')
    if vit:
        num_layers = 12
    else:
        num_layers = 4
    if not distil:
        if vit:
            get_layer_func = partial(get_vit_layer_distil, num_layers=num_layers + 2)
        else:
            get_layer_func = partial(get_mmn_layer, num_layers=num_layers + 2)
    else:
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    
    scales = list(0.65 ** i for i in reversed(range(num_layers + 2)))
    
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        print(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        print(f'No weight decay keywords: {skip_keywords}')

    parameters = get_finetune_param_groups(model, lr, 0.05,
                                           get_layer_func, scales, skip, skip_keywords)

    optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=lr, weight_decay=0.05)

    return optimizer

def get_mmn_layer(name, num_layers):
    if name.startswith("stages"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    elif name.startswith("stem"):
        return 0
    else:
        return num_layers - 1
        
def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1
    
def get_vit_layer_distil(name, num_layers):
    if name in ("encoder.cls_token", "encoder.mask_token", "encoder.pos_embed"):
        return 0
    elif name.startswith("encoder.patch_embed"):
        return 0
    elif name.startswith("encoder.rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("encoder.blocks"):
        layer_id = int(name.split('.')[2])
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin