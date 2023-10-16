import torch.nn as nn
from transformers import AdamW


lr_dict = {
    'bert': {'lr': 1e-5, 'weight_decay': 0.02, 'eps': 1e-6},
    'default': {'lr': 1e-3, 'weight_decay': 0.01, 'eps': 1e-6},
}


def create_optimizer(model: nn.Module):
    # Set learning_rates for each layers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters_decay = []
    optimizer_grouped_parameters_no_decay = []
    group_id = {}

    for i, key in enumerate(lr_dict):
        optimizer_grouped_parameters_decay.append({'params': [],
                                                   'weight_decay': lr_dict[key]['weight_decay'],
                                                   'lr': lr_dict[key]['lr'],
                                                   'eps': lr_dict[key]['eps']})
        optimizer_grouped_parameters_no_decay.append({'params': [],
                                                      'weight_decay': 0.0,
                                                      'lr': lr_dict[key]['lr'],
                                                      'eps': lr_dict[key]['eps']})
        group_id[key] = i

    for n, p in model.named_parameters():
        index = group_id['default']
        for key in lr_dict:
            if key in n:
                index = group_id[key]
                break

        if any(nd in n for nd in no_decay):
            optimizer_grouped_parameters_no_decay[index]['params'].append(p)
        else:
            optimizer_grouped_parameters_decay[index]['params'].append(p)

    optimizer = AdamW(
        optimizer_grouped_parameters_decay + optimizer_grouped_parameters_no_decay,
        lr=lr_dict['default']['lr'],
        eps=lr_dict['default']['eps'],
        )

    return optimizer
