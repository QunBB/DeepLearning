"""
CIKM'2021：One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction

https://arxiv.org/abs/2101.11427
"""
from functools import partial
from typing import List, Callable, Optional, Dict, Type

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

from ..utils.core import dnn_layer, dice
from ..utils.interaction import AttentionBase
from ..utils.type_declaration import Field


class PartitionedNormalization:
    """
    支持一个批次中包含多个场景的样本
    """

    def __init__(self,
                 num_domain,
                 dim,
                 name,
                 **kwargs):
        self.bn_list = [tf.layers.BatchNormalization(center=False, scale=False, name=f"{name}_bn_{i}", **kwargs) for i in
                        range(num_domain)]

        self.global_gamma = tf.get_variable(
            name=f"{name}_global_gamma",
            shape=[dim],
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )
        self.global_beta = tf.get_variable(
            name=f"{name}_global_beta",
            shape=[dim],
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        self.domain_gamma = tf.get_variable(
            name=f"{name}_domain_gamma",
            shape=[len(self.bn_list), dim],
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )
        self.domain_beta = tf.get_variable(
            name=f"{name}_domain_beta",
            shape=[len(self.bn_list), dim],
            initializer=tf.zeros_initializer(),
            trainable=True
        )

    def generate_grid_tensor(self, indices, dim):
        y = tf.range(dim)
        x_grid, y_grid = tf.meshgrid(indices, y)
        return tf.transpose(tf.stack([x_grid, y_grid], axis=-1), [1, 0, 2])

    def __call__(self, inputs, domain_index, training=None):
        domain_index = tf.cast(tf.reshape(domain_index, [-1]), "int32")
        dim = inputs.shape.as_list()[-1]

        output = inputs
        # compute each domain's BN individually
        for i, bn in enumerate(self.bn_list):
            mask = tf.equal(domain_index, i)
            single_bn = self.bn_list[i](tf.boolean_mask(inputs, mask), training=training)
            single_bn = (self.global_gamma + self.domain_gamma[i]) * single_bn + (
                        self.global_beta + self.domain_beta[i])

            # get current domain samples' indices
            indices = tf.boolean_mask(tf.range(tf.shape(inputs)[0]), mask)
            indices = self.generate_grid_tensor(indices, dim)
            output = tf.cond(
                tf.reduce_any(mask),
                lambda: tf.reshape(tf.tensor_scatter_nd_update(output, indices, single_bn), [-1, dim]),
                lambda: output
            )

        return output


class STAR:
    def __init__(self,
                 fields: List[Field],
                 num_domain: int,
                 attention_agg: Type[AttentionBase],
                 star_fcn_input_size: int,
                 star_fcn_units: List[int],
                 aux_net_hidden_units: List[int],
                 aux_net_activation: Callable = dice,
                 aux_net_dropout: Optional[float] = 0.,
                 aux_net_use_bn: Optional[bool] = True,
                 l2_reg: float = 0.,
                 gru_hidden_size: int = 1,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = tf.nn.sigmoid,
                 domain_indicator_field_name: str = 'domain_indicator',
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param num_domain: 场景数量
        :param attention_agg: Attention聚合, 参考DIN,DIEN
        :param star_fcn_input_size: Star Topology FCN的第一层输入size
        :param star_fcn_units: Star Topology FCN的隐藏层size列表
        :param aux_net_hidden_units: 辅助网络的隐藏层size列表
        :param aux_net_activation: 辅助网络激活函数
        :param aux_net_dropout: 辅助网络dropout
        :param aux_net_use_bn: 辅助网络是否使用BN
        :param l2_reg: 正则惩罚项
        :param gru_hidden_size: Attention参数
        :param attention_hidden_units: Attention参数
        :param attention_activation: Attention参数
        :param domain_indicator_field_name: 场景指示器对应的field名称
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """

        self.domain_indicator_field_name = domain_indicator_field_name

        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg)
                                                               )

        assert domain_indicator_field_name in self.embedding_table, f"The field of domain indicator is missing: `{domain_indicator_field_name}`"

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

        with tf.variable_scope(name_or_scope='attention_layer'):
            self.attention_agg = attention_agg(gru_hidden_size, attention_hidden_units, attention_activation)

        self.start_pn = partial(PartitionedNormalization, num_domain=num_domain, name='star_pn')
        self.aux_net_pn = partial(PartitionedNormalization, num_domain=num_domain, name='aux_net_pn')

        with tf.variable_scope(name_or_scope='star_fcn'):
            self.shared_bias = [tf.get_variable(f'star_fcn_b_shared_{i}', shape=[1, star_fcn_units[i]])
                                for i in range(len(star_fcn_units))]
            self.domain_bias = [tf.get_variable(f'star_fcn_b_domain_{i}', shape=[num_domain, star_fcn_units[i]])
                                for i in range(len(star_fcn_units))]

            star_fcn_units.insert(0, star_fcn_input_size)
            self.shared_weights = [tf.get_variable(f'star_fcn_w_shared_{i}', shape=[1, star_fcn_units[i], star_fcn_units[i+1]],
                                                   regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   initializer=init_ops.glorot_normal_initializer(), )
                                   for i in range(len(star_fcn_units) - 1)]
            self.domain_weights = [tf.get_variable(f'star_fcn_w_domain_{i}', shape=[num_domain, star_fcn_units[i] * star_fcn_units[i + 1]],
                                                   regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   initializer=init_ops.glorot_normal_initializer(), )
                                   for i in range(len(star_fcn_units) - 1)]

        with tf.variable_scope(name_or_scope='auxiliary_network'):
            self.auxiliary_network = partial(dnn_layer,
                                             hidden_units=aux_net_hidden_units,
                                             activation=aux_net_activation,
                                             use_bn=aux_net_use_bn,
                                             dropout=aux_net_dropout,
                                             l2_reg=l2_reg)

    def star_fcn(self, inputs, domain_index, layer_index):
        inputs = tf.expand_dims(inputs, axis=1)

        domain_weight = tf.reshape(tf.nn.embedding_lookup(self.domain_weights[layer_index], domain_index),
                                   [-1] + self.shared_weights[layer_index].shape.as_list()[1:])
        weights = self.shared_weights[layer_index] * domain_weight
        domain_bias = tf.reshape(tf.nn.embedding_lookup(self.domain_bias[layer_index], domain_index),
                                   [-1] + self.shared_bias[layer_index].shape.as_list()[1:])
        bias = self.shared_bias[layer_index] + domain_bias

        output = math_ops.matmul(inputs, weights) + tf.expand_dims(bias, axis=1)

        return tf.squeeze(output, axis=1)

    def __call__(self,
                 user_behaviors_ids: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 target_ids: Dict[str, tf.Tensor],
                 other_feature_ids: Dict[str, tf.Tensor],
                 domain_index: tf.Tensor,
                 is_training: bool = True
                 ):
        """

        :param user_behaviors_ids: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id
        :param sequence_length: 用户行为序列长度 [B]
        :param target_ids: 候选ID [B]
        :param other_feature_ids: 其他特征，如用户特征及上下文特征
        :param domain_index: 场景指示器ID，表示当前mini-batch为第n个场景，只取第一个数据来表示当前场景ID
        :param is_training:
        :return:
        """
        # 用户行为历史embedding
        user_behaviors_embeddings = []
        target_embeddings = []
        for name in user_behaviors_ids:
            user_behaviors_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], user_behaviors_ids[name]))
            target_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], target_ids[name]))
        user_behaviors_embeddings = self.func(user_behaviors_embeddings)
        target_embeddings = self.func(target_embeddings)

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], other_feature_ids[name]))
        other_feature_embeddings = array_ops.concat(other_feature_embeddings, axis=-1)

        with tf.variable_scope(name_or_scope='attention_layer'):
            att_outputs = self.attention_agg(user_behaviors_embeddings, target_embeddings, sequence_length)
            if isinstance(att_outputs, (list, tuple)):
                att_outputs = att_outputs[-1]

        domain_index = array_ops.reshape(domain_index, [-1])

        with tf.variable_scope(name_or_scope='partitioned_normalization'):
            agg_inputs = array_ops.concat([att_outputs, target_embeddings, other_feature_embeddings], axis=-1)

            pn_outputs = self.start_pn(dim=agg_inputs.shape.as_list()[-1])(agg_inputs, domain_index=domain_index, training=is_training)

        with tf.variable_scope(name_or_scope='star_fcn'):
            star_fcn_outputs = pn_outputs
            for i in range(len(self.shared_weights)):
                star_fcn_outputs = self.star_fcn(star_fcn_outputs, domain_index, i)

            star_logit = tf.layers.dense(star_fcn_outputs, 1, kernel_initializer=init_ops.glorot_normal_initializer())

        with tf.variable_scope(name_or_scope='auxiliary_network'):
            # 场景指示器
            domain_embedding = tf.nn.embedding_lookup(self.embedding_table[self.domain_indicator_field_name], domain_index)
            aux_inputs = array_ops.concat([array_ops.repeat(array_ops.reshape(domain_embedding, [1, -1]),
                                                     array_ops.shape(agg_inputs)[0], axis=0), agg_inputs], axis=-1)
            aux_inputs = self.aux_net_pn(dim=aux_inputs.shape.as_list()[-1])(aux_inputs, domain_index=domain_index, training=is_training)
            aux_outputs = self.auxiliary_network(aux_inputs, is_training=is_training)

            aux_logit = tf.layers.dense(aux_outputs, 1, kernel_initializer=init_ops.glorot_normal_initializer())

        return array_ops.reshape(tf.nn.sigmoid(star_logit + aux_logit), [-1])
