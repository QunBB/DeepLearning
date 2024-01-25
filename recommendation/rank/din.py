"""
论文：Deep Interest Network for Click-Through Rate Prediction

链接：https://arxiv.org/pdf/1706.06978.pdf

references：https://github.com/zhougr1993/DeepInterestNetwork
"""
from typing import Optional, Dict, List, Callable, Union
import tensorflow as tf
from dataclasses import dataclass
from functools import partial

from ..utils.core import dnn_layer, dice
from ..utils.type_declaration import DINField


class BaseModelMiniBatchReg:
    """
    支持Mini-batch Aware Regularization的模型基类。
    用于序列填充的ID不要使用，否则会影响准确性：
        如`特征ID=0`用于mask，那么其他正常特征ID需从`1`开始
    """
    def __init__(self,
                 fields: List[DINField],
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """
        self.mb_reg_params = []
        self._mba_reg_loss = tf.constant(value=0, dtype=tf.float32)
        self.feature_ids_occurrence = {}
        self.embedding_table = {}
        for field in fields:
            mb_reg = field.mini_batch_regularization and field.ids_occurrence is not None  # 是否使用Mini-batch Aware Regularization
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.embedding_dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg) if not mb_reg and field.l2_reg else None
                                                               )
            if mb_reg:
                assert field.vocabulary_size == len(field.ids_occurrence)
                self.feature_ids_occurrence[field.name] = tf.convert_to_tensor(field.ids_occurrence, dtype=tf.float32)

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

    def embedding_lookup(self,
                         feature_name,
                         ids,
                         padding_id=0,
                         partition_strategy="mod",
                         scope=None,
                         validate_indices=True,  # pylint: disable=unused-argument
                         max_norm=None):
        """
        用于序列填充的ID不要使用，否则会影响准确性。
        如`特征ID=0`用于mask，那么其他正常特征ID需从`1`开始
        :param feature_name: 特征名称
        :param padding_id: 用于填充的ID
        :param ids: 对应tf.nn.embedding_lookup
        :param partition_strategy: 对应tf.nn.embedding_lookup
        :param scope: 对应tf.nn.embedding_lookup的`name`参数
        :param validate_indices: 对应tf.nn.embedding_lookup
        :param max_norm: 对应tf.nn.embedding_lookup
        :return:
        """
        embeddings = tf.nn.embedding_lookup(self.embedding_table[feature_name],
                                            ids, partition_strategy, scope, validate_indices, max_norm)

        if feature_name in self.feature_ids_occurrence:
            # 当前批次出现过的feature id（去重）以及对应的权重参数w_j
            vectorize_ids = tf.reshape(ids, shape=[-1])
            unique_ids, _ = tf.unique(vectorize_ids, out_idx=ids.dtype)
            unique_embeddings = tf.nn.embedding_lookup(self.embedding_table[feature_name],
                                                       unique_ids, partition_strategy, 'unique_' + feature_name,
                                                       validate_indices, max_norm)
            # 获取当前批次中每个feature id的频次
            unique_ids_occurrence = tf.nn.embedding_lookup(self.feature_ids_occurrence[feature_name], unique_ids)
            # 对填充embedding进行mask
            unique_embeddings_mask = unique_embeddings * tf.expand_dims(tf.cast(tf.not_equal(unique_ids, padding_id), tf.float32), axis=1)
            # 计算当前批次出现过的feature id的正则loss
            self.mb_reg_params.append(tf.reduce_sum(tf.norm(unique_embeddings_mask, axis=1) / unique_ids_occurrence))

        return embeddings

    @property
    def mba_reg_loss(self):
        if not self.mb_reg_params:
            return self._mba_reg_loss

        self._mba_reg_loss += tf.add_n(self.mb_reg_params)
        return self._mba_reg_loss


class DIN(BaseModelMiniBatchReg):
    def __init__(self,
                 fields: List[DINField],
                 mlp_hidden_units: List[int],
                 mlp_activation: Callable = dice,
                 mlp_dropout: Optional[float] = 0.,
                 mlp_use_bn: Optional[bool] = True,
                 mlp_l2_reg: float = 0.,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = dice,
                 mode: str = 'concat'):
        super().__init__(fields, mode)
        self.mlp_layer = partial(dnn_layer, hidden_units=mlp_hidden_units, activation=mlp_activation, use_bn=mlp_use_bn,
                                 dropout=mlp_dropout, l2_reg=mlp_l2_reg)
        self.attention_layer = partial(attention, ffn_hidden_units=attention_hidden_units, ffn_activation=attention_activation)

    def __call__(self,
                 user_behaviors_ids: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 target_ids: Dict[str, tf.Tensor],
                 other_feature_ids: Dict[str, tf.Tensor],
                 is_training: bool = True
                 ):
        """

        :param user_behaviors_ids: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id
        :param sequence_length: 用户行为序列长度 [B]
        :param target_ids: 候选ID [B]
        :param other_feature_ids: 其他特征，如用户特征及上下文特征
        :param is_training:
        :return:
        """
        # 将候选ID和用户行为序列的ID进行拼接
        concat_item_embeddings = []  # [B, N+1, D] * m
        for name in user_behaviors_ids:
            # [B, N+1, D]
            embeddings = self.embedding_lookup(feature_name=name,
                                               ids=tf.concat([tf.expand_dims(target_ids[name], axis=1),
                                                              user_behaviors_ids[name]], axis=-1))
            concat_item_embeddings.append(embeddings)
        concat_item_embeddings = self.func(concat_item_embeddings)  # [B, N+1, D']
        target_embedding = concat_item_embeddings[:, 0, :]  # [B, D']
        user_behaviors_embedding = concat_item_embeddings[:, 1:, :]  # [B, N, D']

        with tf.variable_scope(name_or_scope='attention'):
            # [B, D']
            user_behaviors_embedding_with_attention = self.attention_layer(queries=target_embedding,
                                                                           keys=user_behaviors_embedding,
                                                                           keys_length=sequence_length)
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(self.embedding_lookup(name, other_feature_ids[name]))
        other_feature_embeddings = tf.concat(other_feature_embeddings, axis=-1)

        dnn_inputs = tf.concat([user_behaviors_embedding_with_attention, target_embedding, other_feature_embeddings], axis=-1)
        with tf.variable_scope(name_or_scope='mlp_layer'):
            output = self.mlp_layer(inputs=dnn_inputs, is_training=is_training)
            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                     # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                     kernel_initializer=tf.glorot_normal_initializer())
        return tf.reshape(output, [-1])


def attention(queries, keys, keys_length,
              ffn_hidden_units=[80, 40], ffn_activation=dice):
    """

    :param queries: [B, H]
    :param keys: [B, T, H]
    :param keys_length: [B]
    :return: [B, H]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    hidden_layer = dnn_layer(din_all, ffn_hidden_units, ffn_activation, use_bn=False, scope='attention')
    outputs = tf.layers.dense(hidden_layer, 1, activation=None)
    outputs = tf.reshape(outputs, [-1, 1, tf.shape(keys)[1]])
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return tf.squeeze(outputs)


def attention_multi_items(queries, keys, keys_length,
                          ffn_hidden_units=[80, 40],
                          ffn_activation=dice):
    """

    :param queries: [B, N, H] N is the number of ads
    :param keys: [B, T, H]
    :param keys_length:  [B]
    :return: [B, N, H]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries_nums = queries.get_shape().as_list()[1]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    hidden_layer = dnn_layer(din_all, ffn_hidden_units, ffn_activation, use_bn=False, scope='attention')
    outputs = tf.layers.dense(hidden_layer, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    outputs = tf.reshape(outputs, [-1, queries_nums, 1, max_len])
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])  # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])

    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
    return tf.squeeze(outputs)
