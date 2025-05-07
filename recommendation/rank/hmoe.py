"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
from collections import defaultdict
from functools import partial
from typing import Optional, Callable, List, Dict, Any

import tensorflow as tf

from .interaction_expert import Expert
from ..utils.core import dnn_layer


class HMoE:
    def __init__(self,
                 expert_group: Dict[str, Dict[Expert, Dict[str, Any]]],
                 dnn_hidden_units: List[int],
                 gate_weighted: bool = False,
                 sum_weighted: bool = False,
                 dropout: float = 0.,
                 l2_reg: float = 0.,
                 dnn_activation: Optional[Callable] = tf.nn.relu,
                 dnn_use_bn: bool = True,
                 ):
        """Heterogeneous Mixture-of-Experts with Multi-Embedding

        :param expert_group: 每一组专家网络，key是组名，value是专家网络类+参数的字段
        :param dnn_hidden_units: DNN的每一层隐藏层大小
        :param gate_weighted: 是否对多个专家输出进行门控的加权求和
        :param sum_weighted: 是否对多个专家输出进行简单的加权求和
        :param dropout:
        :param l2_reg: 正则惩罚
        :param dnn_activation: 激活函数
        :param dnn_use_bn: 是否使用batch_normalization
        """
        num_experts = 0
        self.expert_group = defaultdict(list)
        # 初始化专家网络模型
        for group in expert_group:
            for expert in expert_group[group]:
                self.expert_group[group].append(
                    expert.init_layer(**expert_group[group][expert])
                )
                num_experts += 1

        self.dnn_layer = partial(dnn_layer, hidden_units=dnn_hidden_units, activation=dnn_activation,
                                 dropout=dropout, use_bn=dnn_use_bn, l2_reg=l2_reg)

        self.gate_weighted = gate_weighted
        self.sum_weighted = sum_weighted
        if sum_weighted:
            self.weights = tf.get_variable("weight", shape=[1, num_experts, 1], initializer=tf.initializers.ones())

    def __call__(self,
                 group_embeddings: Dict[str, List[tf.Tensor]],
                 is_training: bool = True):
        """

        :param group_embeddings: 每一组经过embedding layer的输入，分组要与`expert_group`对应，同一个分组是共享同一个embedding table
        :param is_training:
        :return:
        """
        experts_output = defaultdict(list)
        # 每一组的专家交互
        for group in self.expert_group:
            embeddings = group_embeddings[group]
            for i, expert in enumerate(self.expert_group[group]):
                # 专家网络交互输出
                interaction = expert(embeddings, is_training=is_training)
                # 接一层MLPs
                interaction = self.dnn_layer(interaction, is_training=is_training, scope=f"{group}_expert_{i}")
                experts_output[group].append(interaction)

        final_interaction = []
        for group in group_embeddings:
            final_interaction.extend(experts_output[group])
        # 多个专家聚合
        if self.gate_weighted:  # 门控的加权求和
            scores = []
            for group in group_embeddings:
                scores.append(
                    tf.layers.dense(group_embeddings[group], len(self.expert_group[group]))
                )

            scores = tf.nn.softmax(tf.concat(scores, axis=-1), axis=-1)
            final_interaction = tf.squeeze(
                tf.matmul(tf.expand_dims(scores, axis=1), tf.stack(final_interaction, axis=1)),
                axis=1)
        elif self.sum_weighted:  # 简单的加权求和
            final_interaction = tf.reduce_sum(
                tf.stack(final_interaction, axis=1) * self.weights,
                axis=1
            )
        else:  # 简单的元素位相加
            final_interaction = sum(final_interaction)

        output = tf.layers.dense(final_interaction, 1, activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])
