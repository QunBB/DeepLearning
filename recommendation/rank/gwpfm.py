"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
import tensorflow as tf
from typing import Optional, Callable, List, Union
from typing import Dict as OrderedDictType
from functools import partial
from collections import defaultdict
from itertools import combinations

from ..utils.core import dnn_layer
from ..utils.interaction import LinearEmbedding
from ..utils.type_declaration import Field, LinearTerms


class GwPFM:
    def __init__(self,
                 fields_list: List[Field],
                 dnn_hidden_units: List[int],
                 linear_type: Optional[LinearTerms] = None,
                 add_bias: bool = False,
                 dropout: float = 0.,
                 l2_reg: float = 0.,
                 dnn_activation: Optional[Callable] = None,
                 dnn_use_bn: bool = True,
                 ):
        """GwPFM: Group-weighted Part-aware Factorization Machines

        :param fields_list: 所有fields列表，并使用`field.group`来对fields分组
        :param dnn_hidden_units: DNN的每一层隐藏层大小
        :param linear_type: 线性项，设为`None`则表示不使用线性项
        :param add_bias: 最后的logit是否添加bias
        :param dropout:
        :param l2_reg: 正则惩罚
        :param dnn_activation: 激活函数
        :param dnn_use_bn: 是否使用batch_normalization
        """
        self.num_fields = len(fields_list)

        # 统计分组field
        self.group_dict = defaultdict(list)
        for field in fields_list:
            self.group_dict[field.group].append(field.name)
        self.num_groups = len(self.group_dict)

        for field in fields_list:
            field.dim *= self.num_groups

        self.interaction_strengths = tf.get_variable('interaction_strengths',
                                                     shape=[self.num_groups, self.num_groups],
                                                     initializer=tf.ones_initializer(),
                                                     regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        if linear_type is None:
            self.linear = LinearEmbedding(fields_list)
        else:
            self.linear = LinearEmbedding(fields_list, linear_type)

        self.dnn_layer = partial(dnn_layer, hidden_units=dnn_hidden_units, activation=dnn_activation,
                                 dropout=dropout, use_bn=dnn_use_bn, l2_reg=l2_reg)
        self.l2_reg = l2_reg

        self.add_bias = add_bias
        self.add_linear = linear_type is not None

        self.linear_w = tf.get_variable('linear_w', shape=[1], initializer=tf.ones_initializer())
        self.dnn_w = tf.get_variable('dnn_w', shape=[1], initializer=tf.ones_initializer())
        self.bias = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 context_embeddings: Optional[tf.Tensor] = None,
                 is_training: bool = True
                 ):
        """

        :param sparse_inputs_dict: 离散特征，经过LabelEncoder之后的输入(未经过embedding layer的输入)
        :param dense_inputs_dict: 连续值特征(未经过embedding layer的输入)
        :param context_embeddings: 其他经过embedding layer的输入，如历史行为交互表征
        :param is_training:
        :return:
        """
        embeddings_dict, linear_logit = self.linear(sparse_inputs_dict, dense_inputs_dict, as_dict=True)

        # 将同一个组的embeddings放置在一块，并记录其对应的分组ID
        inputs = []
        field_group_idx = []
        for i, group in enumerate(self.group_dict.values()):
            for name in group:
                inputs.append(embeddings_dict[name])
                field_group_idx.append(i)

        inputs = tf.stack(inputs, axis=1)
        inputs = tf.reshape(inputs, [-1, len(field_group_idx), self.num_groups, inputs.shape[-1] // self.num_groups])

        interaction = self._interaction(inputs, field_group_idx)

        if context_embeddings is not None:
            interaction = tf.concat([interaction, context_embeddings], axis=-1)

        dnn_output = self.dnn_layer(interaction, is_training=is_training)
        dnn_logit = tf.layers.dense(dnn_output, 1,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                    kernel_initializer=tf.glorot_normal_initializer())
        dnn_logit = tf.reshape(dnn_logit, [-1])

        if self.add_linear:
            final_logit = self.dnn_w * dnn_logit + self.linear_w * linear_logit
        else:
            final_logit = dnn_logit

        if self.add_bias:
            final_logit += self.bias

        return tf.nn.sigmoid(final_logit)

    def _interaction(self, inputs, field_group_idx):
        interactions = []
        for i, j in combinations(range(self.num_fields), 2):
            v_i = inputs[:, i, field_group_idx[j], :]
            v_j = inputs[:, j, field_group_idx[i], :]

            interactions.append(v_i * v_j * self.interaction_strengths[field_group_idx[i], field_group_idx[j]])

        return sum(interactions)