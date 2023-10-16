"""
> 论文：Neural Factorization Machines for Sparse Predictive Analytics
>
> 地址：https://arxiv.org/pdf/1708.05027.pdf
"""
from functools import partial
from typing import Dict as OrderedDictType
from typing import List, Callable

import tensorflow as tf

from ..utils.core import dnn_layer
from ..utils.interaction import LinearEmbedding
from ..utils.type_declaration import LinearTerms, Field


class NFM:
    def __init__(self,
                 fields_list: List[Field],
                 dnn_hidden_size: List[int],
                 dnn_activation: Callable = None,
                 dnn_dropout: float = 0.,
                 dnn_use_bn: bool = True,
                 dnn_l2_reg: float = 0.,
                 linear_type: LinearTerms = LinearTerms.LW):
        self.num_fields = len(fields_list)

        self.dnn_layer = partial(dnn_layer, hidden_size=dnn_hidden_size, activation=dnn_activation,
                                 dropout=dnn_dropout, use_bn=dnn_use_bn, l2_reg=dnn_l2_reg)
        self.dnn_l2_reg = dnn_l2_reg

        self.linear = LinearEmbedding(fields_list, linear_type)

        self.global_w = tf.get_variable('global_w', shape=[1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 is_training: bool = True):
        """
        未经过embedding layer的输入
        :param sparse_inputs_dict: 离散特征，经过LabelEncoder之后的输入
        :param dense_inputs_dict: 连续值特征
        :return:
        """
        embeddings, linear_logit = self.linear(sparse_inputs_dict, dense_inputs_dict)

        bi_interaction_output = self._bi_interaction_layer(embeddings)

        dnn_output = self.dnn_layer(bi_interaction_output, is_training=is_training)

        dnn_output = tf.layers.dense(dnn_output, 1,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                     kernel_initializer=tf.glorot_normal_initializer())
        dnn_output = tf.reshape(dnn_output, [-1])

        final_logit = tf.nn.sigmoid(linear_logit + dnn_output + self.global_w)

        return final_logit

    def _bi_interaction_layer(self, interactions):
        interactions = tf.stack(interactions, axis=1)
        square_of_sum = tf.square(tf.reduce_sum(
            interactions, axis=1))
        sum_of_square = tf.reduce_sum(
            interactions * interactions, axis=1)
        fm_logit = square_of_sum - sum_of_square

        return 0.5 * fm_logit
