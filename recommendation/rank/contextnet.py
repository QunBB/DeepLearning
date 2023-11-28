"""
论文：ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding

地址：https://arxiv.org/pdf/2107.1202
"""
import tensorflow as tf
from typing import List, Union
from functools import partial

from ..utils.core import dnn_layer


class ContextNet:
    def __init__(self,
                 num_block: int,
                 agg_dim: int,
                 ffn_type: str,
                 embedding_ln: bool = True,
                 l2_reg: float = 0.,
                 dropout: float = 0.
                 ):
        """

        :param num_block: ContextNet Block的层数
        :param agg_dim: Contextual Embedding中的Aggregation模块的输出维度
        :param ffn_type: ContextNet Block使用Point-wise FFN或者Single-layer FFN
        :param l2_reg:
        :param dropout:
        """
        self.num_block = num_block
        self.agg_dim = agg_dim
        self.embedding_ln = embedding_ln
        self.l2_reg = l2_reg

        self.dnn_layer = partial(dnn_layer, use_bias=False, dropout=dropout, use_bn=False, l2_reg=l2_reg)

        if ffn_type == 'PFFN':
            self.ffn_func = self.point_wise
        elif ffn_type == 'FFN':
            self.ffn_func = self.single_layer
        else:
            raise TypeError('ffn_type only support: "PFFN" or "FFN"')

    def __call__(self,
                 inputs: Union[List[tf.Tensor], tf.Tensor],
                 is_training: bool = True):
        """

        :param inputs: [bs, num_feature, dim] or list of [bs, dim]
        :param is_training:
        :return:
        """
        if isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=1)

        assert len(inputs.shape) == 3

        if self.embedding_ln:
            inputs = tf.contrib.layers.layer_norm(inputs=inputs,
                                                  begin_norm_axis=-1,
                                                  begin_params_axis=-1)

        # stack ContextNet block
        output = inputs
        for _ in range(self.num_block):
            output = self.contextnet_block(output, is_training)

        # flatten
        output = tf.layers.flatten(output)
        output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())
        return tf.reshape(output, [-1])

    def get_contextual_embedding(self, embeddings, is_training):
        shape = embeddings.shape.as_list()
        num_feature = shape[1]
        dim = shape[2]

        # 所有特征共享Aggregation参数
        agg = self.dnn_layer(embeddings, self.agg_dim, activation=tf.nn.relu, is_training=is_training)

        # Project参数不共享
        agg = tf.reshape(agg, [-1, num_feature * self.agg_dim])
        project = self.dnn_layer(agg, num_feature * dim, is_training=is_training)
        return tf.reshape(project, [-1, num_feature, dim])

    def contextnet_block(self, embeddings, is_training):
        contextual_embedding = self.get_contextual_embedding(embeddings, is_training)
        merge = contextual_embedding * embeddings
        output = self.ffn_func(merge, is_training)
        return output

    def point_wise(self, inputs, is_training):
        dim = inputs.shape.as_list()[-1]
        # 整个网络参数共享
        with tf.variable_scope('point-wise', reuse=tf.AUTO_REUSE):
            ffn1 = self.dnn_layer(inputs, dim, activation=tf.nn.relu, scope='ffn1', is_training=is_training)
            ffn2 = self.dnn_layer(ffn1, dim, scope='ffn2', is_training=is_training)
            output = tf.contrib.layers.layer_norm(inputs=ffn2,
                                                  begin_norm_axis=-1,
                                                  begin_params_axis=-1,
                                                  scope='point-wise-ln')

        return output + inputs

    def single_layer(self, inputs, is_training):
        dim = inputs.shape.as_list()[-1]
        # 整个网络参数共享
        with tf.variable_scope('single-layer', reuse=tf.AUTO_REUSE):
            ffn = self.dnn_layer(inputs, dim, scope='ffn', is_training=is_training)
            output = tf.contrib.layers.layer_norm(inputs=ffn,
                                                  begin_norm_axis=-1,
                                                  begin_params_axis=-1,
                                                  scope='single-layer-ln')

        return output
