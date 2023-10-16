"""
论文：Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction

地址：https://arxiv.org/pdf/1601.02376.pdf
"""
from typing import Dict as OrderedDictType
from typing import List, Union

import tensorflow as tf
from dataclasses import dataclass

from ..utils.core import dnn_layer
from ..utils.train_utils import get_assignment_map_from_checkpoint


@dataclass
class Field:
    name: str
    vocabulary_size: int = 1  # dense类型为1


class FNN:
    def __init__(self,
                 fields_list: List[Field],
                 embedding_dim: int,
                 dnn_hidden_size: List[int],
                 dropout: float = 0.,
                 l2_reg: float = 0.,
                 use_bn: bool = False,
                 dnn_l2_reg: float = 0.,
                 fms_checkpoint: str = None
                 ):
        """

        :param fields_list:
        :param embedding_dim: 特征向量的dim
        :param dnn_hidden_size: 隐藏层的size列表
        :param dropout:
        :param l2_reg: 特征向量的l2正则项力度
        :param use_bn: 全连接层是否使用batch_normalization
        :param dnn_l2_reg: 全连接层的权重l2正则项力度
        :param fms_checkpoint: 预训练的FMs模型路径
        """
        self.embeddings_table = {}
        self.weights_table = {}
        self.num_fields = len(fields_list)

        for field in fields_list:
            # embeddings 隐向量
            self.embeddings_table[field.name] = tf.get_variable('emb_' + field.name,
                                                                shape=[field.vocabulary_size, embedding_dim],
                                                                regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

            # 线性项权重
            self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[field.vocabulary_size],
                                                             regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        self.dnn_hidden_size = dnn_hidden_size
        self.dropout = dropout
        self.dnn_l2_reg = dnn_l2_reg
        self.use_bn = use_bn

        # 加载预训练的FMs模型向量
        if fms_checkpoint:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, fms_checkpoint)
            tf.train.init_from_checkpoint(fms_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

    def _get_linear_logit(self,
                          w: tf.Tensor,
                          i: Union[int, tf.Tensor] = 0,
                          x: Union[int, tf.Tensor] = 1,
                          ):
        """线性项计算"""

        return tf.gather(w, i) * x

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 is_training: bool = True):
        dense_real_layer = []

        # 类别特征: (w, v)
        for name, x in sparse_inputs_dict.items():
            v = tf.nn.embedding_lookup(self.embeddings_table[name], x)
            w = self._get_linear_logit(w=self.weights_table[name], i=x)
            dense_real_layer.append(tf.concat([tf.expand_dims(w, axis=1), v], axis=1))

        # 数值特征: v * w
        for name, x in dense_inputs_dict.items():
            v = tf.reshape(self.embeddings_table[name][0], [1, -1])
            w = self._get_linear_logit(w=self.weights_table[name], x=x)
            dense_real_layer.append(tf.expand_dims(w, axis=1) * v)

        dense_real_layer = tf.concat(dense_real_layer, axis=1)

        output = dnn_layer(dense_real_layer, self.dnn_hidden_size, activation=tf.nn.tanh,
                           is_training=is_training, use_bn=self.use_bn, l2_reg=self.dnn_l2_reg, dropout=self.dropout)

        output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])
