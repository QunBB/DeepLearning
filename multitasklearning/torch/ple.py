import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Dict

from utils import DNN


class PLE(nn.Module):
    """One Level PLE.

    :param inputs_dim: Dimension of the inputs.
    :param tower_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param labels_dict: dict. The number of Labels
    :param num_shared_experts: int. The number of Shared Experts
    :param num_task_experts: int. The number of every task Specific Experts
    :param expert_hidden_units: list of positive integer, the layer number and units in each expert layer.
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
                 num_shared_experts: int,
                 num_task_experts: int,
                 expert_hidden_units: List[int],
                 tower_hidden_units=(256, 128),
                 l2_reg_dnn=0., init_std=0.0001,
                 dnn_dropout=0., dnn_activation='relu', dnn_use_bn=False):
        super(PLE, self).__init__()

        self.labels_dict = labels_dict

        self.shared_experts_dnn = nn.ModuleList([DNN(inputs_dim, expert_hidden_units,
                                                     activation=dnn_activation, l2_reg=l2_reg_dnn,
                                                     dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std,
                                                     ) for _ in range(num_shared_experts)])

        self.gate_network = nn.ModuleList([GateNetwork(inputs_dim, num_task_experts, self.shared_experts_dnn,
                                                       expert_hidden_units, l2_reg_dnn, init_std,
                                                       dnn_dropout, dnn_activation, dnn_use_bn
                                                       ) for _ in labels_dict])

        self.task_tower = nn.ModuleList([DNN(expert_hidden_units[-1], tower_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std) for _ in labels_dict])
        self.task_dense = nn.ModuleList(
            [nn.Linear(tower_hidden_units[-1], labels_dict[name], bias=False) for name in labels_dict])

    def forward(self, dnn_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = OrderedDict()
        for index, name in enumerate(self.labels_dict):
            tower_inputs = self.gate_network[index](dnn_inputs)
            tower_output = self.task_tower[index](tower_inputs)
            task_output = self.task_dense[index](tower_output)

            outputs[name] = torch.softmax(task_output, dim=-1)

        return outputs


class GateNetwork(nn.Module):
    def __init__(self, inputs_dim, num_experts, shared_experts_dnn,
                 dnn_hidden_units=(256, 128),
                 l2_reg_dnn=0., init_std=0.0001,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False
                 ):
        super(GateNetwork, self).__init__()
        self.task_experts_dnn = nn.ModuleList([DNN(inputs_dim, dnn_hidden_units,
                                                   activation=dnn_activation, l2_reg=l2_reg_dnn,
                                                   dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                   init_std=init_std) for _ in range(num_experts)])

        self.gates = nn.Linear(inputs_dim, num_experts + len(shared_experts_dnn), bias=False)
        self.shared_experts_dnn = shared_experts_dnn

    def forward(self, inputs):
        experts_output = []
        for dnn in self.shared_experts_dnn:
            experts_output.append(dnn(inputs))

        for dnn in self.task_experts_dnn:
            experts_output.append(dnn(inputs))

        experts_output = torch.stack(experts_output, dim=1)

        gate_weight = self.gates(inputs)
        gate_weight = torch.softmax(gate_weight, dim=-1)
        gate_weight = torch.unsqueeze(gate_weight, dim=2)

        return torch.sum(experts_output * gate_weight, dim=1)


if __name__ == '__main__':
    import numpy as np

    model = PLE(inputs_dim=2056,
                labels_dict={"click": 2, "like": 2},
                num_shared_experts=2,
                num_task_experts=2,
                expert_hidden_units=[256])

    outputs = model(torch.FloatTensor(np.random.random([64, 2056])))

    print(outputs)
    for name in outputs:
        print(name, outputs[name].shape)
