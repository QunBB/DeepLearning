import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Dict

from utils import DNN


class MMoe(nn.Module):
    """One Level MMoe.

    :param inputs_dim: Dimension of the inputs. e.g. {"click": 2, "like": 2}
    :param labels_dict: dict. The number of Labels
    :param num_experts: int. The number of Shared Experts
    :param expert_hidden_units: list of positive integer, the layer number and units in each expert layer.
    :param tower_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param tower_hidden_units: list of positive integer, the layer number and units in each tower layer.
    :param init_std: float,to use as the initializer std of embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether to use BatchNormalization before activation or not in DNN
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 inputs_dim: int,
                 labels_dict: Dict[str, int],
                 num_experts: int,
                 expert_hidden_units: List[int],
                 tower_hidden_units=(256, 128),
                 l2_reg_dnn=0., init_std=0.0001,
                 dnn_dropout=0., dnn_activation='relu', dnn_use_bn=False):
        super(MMoe, self).__init__()

        self.labels_dict = labels_dict

        self.experts_dnn = nn.ModuleList([DNN(inputs_dim, expert_hidden_units,
                                              activation=dnn_activation, l2_reg=l2_reg_dnn,
                                              dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std,
                                              ) for _ in range(num_experts)])
        self.gate_dnn = nn.ModuleList([DNN(inputs_dim, [num_experts],
                                           activation=None, l2_reg=l2_reg_dnn,
                                           use_bn=dnn_use_bn, init_std=init_std,
                                           ) for _ in range(num_experts)])

        self.task_tower = nn.ModuleList([DNN(expert_hidden_units[-1], tower_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std) for _ in labels_dict])
        self.task_dense = nn.ModuleList(
            [nn.Linear(tower_hidden_units[-1], labels_dict[name], bias=False) for name in labels_dict])

    def forward(self, dnn_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()

        experts_output = []
        for dnn in self.experts_dnn:
            experts_output.append(dnn(dnn_inputs))

        for index, name in enumerate(self.labels_dict):
            gate = self.gate_dnn[index](dnn_inputs)
            tower_inputs = _merge_experts_with_gate(experts_output, gate)
            tower_output = self.task_tower[index](tower_inputs)
            task_output = self.task_dense[index](tower_output)
            outputs[name] = torch.softmax(task_output, dim=-1)

        return outputs


def _merge_experts_with_gate(experts: List[torch.Tensor],
                             gate: torch.Tensor):
    experts = torch.stack(experts, dim=1)

    gate_weight = torch.softmax(gate, dim=-1)
    gate_weight = torch.unsqueeze(gate_weight, dim=2)

    return torch.sum(experts * gate_weight, dim=1)


if __name__ == '__main__':
    import numpy as np

    model = MMoe(inputs_dim=2056,
                 labels_dict={"click": 2, "like": 2},
                 num_experts=2,
                 expert_hidden_units=[256])

    outputs = model(torch.FloatTensor(np.random.random([64, 2056])))

    print(outputs)
    for name in outputs:
        print(name, outputs[name].shape)
