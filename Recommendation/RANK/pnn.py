"""
新
    论文：Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data
    地址：https://arxiv.org/pdf/1807.00311.pdf
旧
    论文：Product-based Neural Networks for User Response Prediction
    地址：https://arxiv.org/pdf/1611.00144.pdf
"""
from typing import List, Union, Callable
from enum import IntEnum
import tensorflow as tf

from ..Utils.core import dnn_layer


class KernelType(IntEnum):
    """
    0-2对应KPNN不同kernel形式，3对应PIN的micro net
    """
    Num = 0
    Vec = 1
    Mat = 2
    Net = 3  # PIN


class PNN:
    """
    `kernel_type`不为None时，使用新论文的product操作
    `kernel_type`为None时，则使用旧论文的product操作
    """
    def __init__(self,
                 num_fields: int,
                 dim: int,
                 dnn_hidden_size: List[int],
                 add_inner_product: bool,
                 add_outer_product: bool,
                 kernel_type: Union[None, KernelType] = None,
                 micro_net_size: Union[None, int] = None,
                 micro_net_activation: Union[None, Callable] = None,
                 product_layer_size: int = 0,
                 dnn_l2_reg: float = 0.,
                 emb_l2_reg: float = 0.,
                 dropout: float = 0.,
                 use_bn: bool = True):
        """

        :param num_fields:
        :param dim: 特征embeddings的维度大小dim
        :param dnn_hidden_size: DNN的隐藏层size
        :param add_inner_product:
        :param add_outer_product:
        :param kernel_type:
        :param micro_net_size: 新论文PIN的micro net中间层size
        :param micro_net_activation: 新论文PIN的micro net中间层激活函数
        :param product_layer_size: 旧论文的product输出size
        :param dnn_l2_reg:
        :param emb_l2_reg:
        :param dropout:
        :param use_bn:
        """
        self.add_inner_product = add_inner_product
        self.add_outer_product = add_outer_product
        self.product_layer_dim = product_layer_size
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_l2_reg = dnn_l2_reg
        self.dropout = dropout
        self.use_bn = use_bn

        self.kernel_type = kernel_type
        if kernel_type is not None:
            # 新论文的product

            num_pairs = int(num_fields * (num_fields - 1) / 2)

            if self.kernel_type == KernelType.Net:
                # PIN
                self.micro_net_activation = micro_net_activation

                self.kernel_w = [tf.get_variable('micro_net_w1', shape=[1, num_pairs, dim*3, micro_net_size],
                                                 initializer=tf.glorot_normal_initializer()),
                                 tf.get_variable('micro_net_w2', shape=[1, num_pairs, micro_net_size, 1],
                                                 initializer=tf.glorot_normal_initializer())]
                self.kernel_b = [tf.get_variable('micro_net_b1', shape=[1, num_pairs, micro_net_size],
                                                 initializer=tf.glorot_normal_initializer()),
                                 tf.get_variable('micro_net_b2', shape=[1, num_pairs, 1],
                                                 initializer=tf.glorot_normal_initializer())]

            # 不同kernel的KPNN
            elif self.kernel_type == KernelType.Mat:
                self.kernel = tf.get_variable('product_kernel', shape=[dim, num_pairs, dim],
                                              initializer=tf.glorot_normal_initializer())
            elif self.kernel_type == KernelType.Vec:
                self.kernel = tf.get_variable('product_kernel', shape=[num_pairs, dim],
                                              initializer=tf.glorot_normal_initializer())
            elif self.kernel_type == KernelType.Num:
                self.kernel = tf.get_variable('product_kernel', shape=[num_pairs, 1],
                                              initializer=tf.glorot_normal_initializer())
            else:
                raise TypeError('not support such kernel')

            self.inner_product_func = self._new_inner_product
            self.outer_product_func = self._new_outer_product
        else:
            # 旧论文的product

            # inner product layer
            self.theta = tf.get_variable('delta', shape=[1, product_layer_size, num_fields, dim],
                                         initializer=tf.glorot_normal_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(emb_l2_reg))
            # self.inner_bias = tf.get_variable('inner_bias', shape=[1, product_layer_dim], initializer=tf.zeros_initializer())

            # outer product layer
            self.outer_product_weight = tf.get_variable('outer_product_weight',
                                                        shape=[1, product_layer_size, dim, dim],
                                                        initializer=tf.glorot_normal_initializer(),
                                                        regularizer=tf.contrib.layers.l2_regularizer(emb_l2_reg))
            # self.outer_bias = tf.get_variable('outer_bias', shape=[1, product_layer_dim], initializer=tf.zeros_initializer())

            self.inner_product_func = self._inner_product
            self.outer_product_func = self._outer_product

    def __call__(self, embeddings_list, is_training=True):
        l_z = tf.concat(embeddings_list, axis=1)

        if self.kernel_type is None:
            l_z = tf.layers.dense(l_z, self.product_layer_dim,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                  kernel_initializer=tf.glorot_normal_initializer())

        dnn_inputs = [l_z]

        if self.add_inner_product:
            l_p = self.inner_product_func(embeddings_list)
            dnn_inputs.append(l_p)
        if self.add_outer_product:
            l_p = self.outer_product_func(embeddings_list)
            dnn_inputs.append(l_p)

        dnn_inputs = tf.concat(dnn_inputs, axis=-1)

        dnn_output = dnn_layer(dnn_inputs, self.dnn_hidden_size, activation=tf.nn.relu,
                               dropout=self.dropout, is_training=is_training, use_bn=self.use_bn, l2_reg=self.dnn_l2_reg)

        output = tf.layers.dense(dnn_output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])

    def _inner_product(self, embeddings_list):
        # [bs, nums_field, dim]
        embeddings = tf.stack(embeddings_list, axis=1)
        # [bs, 1, nums_field, dim]
        embeddings = tf.expand_dims(embeddings, axis=1)

        # [bs, product_layer_dim, nums_field, dim]
        delta = tf.multiply(embeddings, self.theta)

        # [bs, product_layer_dim, dim]
        l_p = tf.reduce_sum(delta, axis=2)

        # [bs, product_layer_dim]
        l_p = tf.reduce_sum(tf.pow(l_p, 2), axis=2)

        return l_p

    def _outer_product(self, embeddings_list):
        # [bs, nums_field, dim]
        embeddings = tf.stack(embeddings_list, axis=1)
        # [bs, dim]
        sum_embeddings = tf.reduce_sum(embeddings, axis=1)
        # [bs, dim, 1]
        sum_embeddings = tf.expand_dims(sum_embeddings, axis=2)
        # [bs, dim, dim]
        p = tf.matmul(sum_embeddings, sum_embeddings, transpose_b=True)
        # [bs, product_layer_dim, dim, dim]
        l_p = self.outer_product_weight * tf.expand_dims(p, axis=1)
        # [bs, product_layer_dim]
        l_p = tf.reduce_sum(l_p, axis=[2, 3])
        return l_p

    def _get_embedding_pairs(self, embeddings_list):
        num_fields = len(embeddings_list)

        p = []
        q = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                p.append(embeddings_list[i])
                q.append(embeddings_list[j])
        # [bs, paris, dim]
        p = tf.stack(p, axis=1)
        q = tf.stack(q, axis=1)

        return p, q

    def _get_mirco_embedding_paris(self, embeddings_list):
        num_fields = len(embeddings_list)
        p = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                p.append(tf.concat([embeddings_list[i], embeddings_list[j], embeddings_list[i] * embeddings_list[j]], axis=-1))

        # [bs, paris, 3*dim]
        p = tf.stack(p, axis=1)
        return p

    def _new_inner_product(self, embeddings_list):
        p, q = self._get_embedding_pairs(embeddings_list)

        ip = tf.reduce_sum(p * q, [-1])

        return ip

    def _new_outer_product(self, embeddings_list):
        # PIN: micro network kernel
        if self.kernel_type == KernelType.Net:
            # [bs, paris, 3*dim]
            p = self._get_mirco_embedding_paris(embeddings_list)
            # [bs, paris, 3*dim, 1]
            p = tf.expand_dims(p, axis=-1)

            # [bs, paris, size]
            sub_net_1 = tf.reduce_sum(
                # [bs, paris, 3*dim, size]
                tf.multiply(p, self.kernel_w[0]),
                axis=2
            ) + self.kernel_b[0]

            if self.micro_net_activation is not None:
                sub_net_1 = self.micro_net_activation(sub_net_1)

            # [bs, paris, size, 1]
            sub_net_1 = tf.expand_dims(sub_net_1, axis=-1)
            # [bs, paris, 1]
            sub_net_2 = tf.reduce_sum(
                # [bs, paris, size, 1]
                tf.multiply(sub_net_1, self.kernel_w[1]),
                axis=2
            ) + self.kernel_b[1]

            return tf.squeeze(sub_net_2, axis=-1)

        # KPNN: linear kernel
        p, q = self._get_embedding_pairs(embeddings_list)

        if self.kernel_type == KernelType.Mat:
            # batch * 1 * pair * k
            p = tf.expand_dims(p, 1)
            # batch * pair
            kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                p, self.kernel),
                            -1),
                        [0, 2, 1]),
                    q),
                -1)
        else:
            # 1 * pair * (k or 1)
            k = tf.expand_dims(self.kernel, 0)
            # batch * pair
            kp = tf.reduce_sum(p * q * k, -1)

        return kp
