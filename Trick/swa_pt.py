import torch
import torch.nn as nn


def apply_swa(model: nn.Module,
              checkpoint_list: list,
              weight_list: list,
              strict: bool = True):
    """

    :param model:
    :param checkpoint_list: 要进行swa的模型路径列表
    :param weight_list: 每个模型对应的权重
    :param strict: 输入模型参数与checkpoint是否需要完全匹配
    :return:
    """

    checkpoint_tensor_list = [torch.load(f, map_location='cpu') for f in checkpoint_list]

    for name, param in model.named_parameters():
        try:
            param.data = sum([ckpt['model'][name] * w for ckpt, w in zip(checkpoint_tensor_list, weight_list)])
        except KeyError:
            if strict:
                raise KeyError(f"Can't match '{name}' from checkpoint")
            else:
                print(f"Can't match '{name}' from checkpoint")

    return model
