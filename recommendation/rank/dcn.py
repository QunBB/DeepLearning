"""
论文：Deep & Cross Network for Ad Click Predictions

地址：https://arxiv.org/pdf/1708.05123.pdf

论文：DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

地址：https://arxiv.org/pdf/2008.13535.pdf
"""
import tensorflow as tf
from typing import Optional, Callable, List
from functools import partial

from ..utils.core import dnn_layer


class DCN:
    def __init__(self,
                 input_dim: int,
                 dnn_hidden_size: List[int],
                 cross_layer_num: int,
                 cross_network_type: str,
                 low_rank_dim: Optional[int] = None,
                 cross_network_activation: Optional[Callable] = None,
                 cross_network_l2_reg: float = 0.,
                 dnn_activation: Optional[Callable] = None,
                 dnn_dropout: float = 0.,
                 dnn_use_bn: bool = True,
                 dnn_l2_reg: float = 0.
                 ):
        self.cross_layer = CrossNetwork(input_dim, cross_layer_num, cross_network_type, low_rank_dim,
                                        cross_network_activation, cross_network_l2_reg)

        self.dnn_layer = partial(dnn_layer, hidden_size=dnn_hidden_size, activation=dnn_activation,
                                 dropout=dnn_dropout, use_bn=dnn_use_bn, l2_reg=dnn_l2_reg)
        self.dnn_l2_reg = dnn_l2_reg

    def __call__(self, inputs: tf.Tensor,
                 is_training: bool = True):

        cross_output = self.cross_layer(inputs)
        dnn_output = self.dnn_layer(inputs=inputs, is_training=is_training)

        stack = tf.concat([cross_output, dnn_output], axis=1)

        output = tf.layers.dense(stack, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])


class CrossNetwork:
    def __init__(self,
                 input_dim: int,
                 layer_num: int,
                 cross_type: str = 'vector',
                 low_rank_dim: Optional[int] = None,
                 activation: Optional[Callable] = None,
                 l2_reg: float = 0.):
        """

        :param input_dim: 输入拼接后的维度
        :param layer_num: cross network的层数
        :param cross_type:  cross weight类型，vector对应DCN，matrix对应DCN-V2
        :param low_rank_dim: 使用low-rank，中间降维维度
        :param activation: 激活函数
        :param l2_reg:
        :return:
        """
        assert cross_type in ['matrix', 'vector']

        self.input_dim = input_dim
        self.layer_num = layer_num
        self.cross_type = cross_type
        self.low_rank_dim = low_rank_dim
        self.activation = activation
        self.l2_reg = l2_reg

        self.bias = [tf.get_variable(f'cross_bias_{i}', [1, input_dim], initializer=tf.zeros_initializer())
                     for i in range(layer_num)]

    def __call__(self, inputs: tf.Tensor):
        assert inputs.shape.as_list()[-1] == self.input_dim and inputs.shape.ndims == 2, \
            "the dimension of inputs must be equal to `input_dim` and rank muse be 2"

        x_0 = inputs
        x_l = inputs

        for i in range(self.layer_num):

            if self.cross_type == 'matrix' and self.low_rank_dim is not None:
                prod_output = tf.layers.dense(x_l, self.low_rank_dim, activation=self.activation,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                              kernel_initializer=tf.glorot_normal_initializer()
                                              )
                prod_output = tf.layers.dense(prod_output, self.input_dim, activation=self.activation,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                              kernel_initializer=tf.glorot_normal_initializer()
                                              )
            else:
                if self.cross_type == 'vector':
                    unit = 1
                else:
                    unit = self.input_dim
                prod_output = tf.layers.dense(x_l, unit, activation=self.activation,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                              kernel_initializer=tf.glorot_normal_initializer()
                                              )

            x_l = x_0 * prod_output + self.bias[i] + x_l

        return x_l

