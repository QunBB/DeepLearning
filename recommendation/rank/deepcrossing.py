"""
论文：Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features

地址：https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf
"""
from typing import List, Union
import tensorflow as tf

from ..utils.core import dnn_layer


class DeepCrossing:
    def __init__(self,
                 residual_size: List[int],
                 l2_reg: float = 0.,
                 dropout: float = 0.,
                 use_bn: bool = True
                 ):
        """

        :param residual_size: 每一层残差网络的中间层size
        :param l2_reg:
        :param dropout:
        :param use_bn:
        """
        self.residual_size = residual_size
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.use_bn = use_bn

    def __call__(self,
                 embeddings: Union[List[tf.Tensor], tf.Tensor],
                 is_training: bool = True):
        if isinstance(embeddings, list):
            embeddings = tf.concat(embeddings, axis=-1)

        if embeddings.shape.ndims != 2:
            raise ValueError('Input tensor must have rank 2')

        residual_output = embeddings
        for size in self.residual_size:
            residual_output = self.residual_layer(residual_output, size, is_training)

        output = tf.layers.dense(residual_output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])

    def residual_layer(self, inputs, hidden_units, is_training):
        dim = inputs.shape.as_list()[-1]
        layer_output = dnn_layer(inputs, hidden_units=[hidden_units, dim],
                                 dropout=self.dropout,
                                 activation=tf.nn.relu,
                                 use_bn=self.use_bn,
                                 l2_reg=self.l2_reg,
                                 is_training=is_training)
        return inputs + layer_output
