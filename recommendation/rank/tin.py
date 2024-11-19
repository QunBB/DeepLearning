"""
Temporal Interest Network for User Response Prediction

WWW'2024：https://arxiv.org/abs/2308.08487
"""
from typing import Optional, Dict, List, Callable, Union
from functools import partial
import tensorflow as tf

from ..utils.transformer import attention_layer, create_attention_mask_from_input_mask
from ..utils.core import dnn_layer
from ..utils.type_declaration import Field


class TIN:
    def __init__(self,
                 fields: List[Field],
                 num_attention_heads: int,
                 attention_head_size: int,
                 hidden_units: List[int],
                 activation: Optional[Union[Callable, str]] = "relu",
                 dropout: Optional[float] = 0.,
                 use_bn: Optional[bool] = True,
                 l2_reg: float = 0.,
                 mode: str = "concat"
                 ):
        """

        :param num_attention_heads: multi-head attention中的注意力头的数量
        :param attention_head_size: multi-head attention中的注意力输出维度
        """
        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg)
                                                               )

        self.position_embeddings = partial(tf.get_variable,
                                           name="position_embeddings",
                                           initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.mlp_layer = partial(dnn_layer, hidden_units=hidden_units, activation=activation, use_bn=use_bn,
                                 dropout=dropout, l2_reg=l2_reg)

        self.attention_layer = partial(attention_layer,
                                       num_attention_heads=num_attention_heads,
                                       size_per_head=attention_head_size,
                                       target_aware=True
                                       )
        self.l2_reg = l2_reg

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

    def merge_position_embedding(self, query, key):
        query = tf.expand_dims(query, axis=1)

        table = self.position_embeddings(shape=[key.shape[1]+1, key.shape[-1]])
        key_pos_emb = tf.nn.embedding_lookup(table, tf.ones_like(key[:, :, 0], dtype="int32") * tf.range(1, key.shape[1]+1))
        query_pos_emb = tf.nn.embedding_lookup(table, tf.ones_like(query[:, :, 0], dtype="int32") * 0)

        return query + query_pos_emb, key + key_pos_emb


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
        # 用户行为历史embedding
        user_behaviors_embeddings = []
        target_embeddings = []
        for name in user_behaviors_ids:
            user_behaviors_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], user_behaviors_ids[name]))
            target_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], target_ids[name]))
        user_behaviors_embeddings = self.func(user_behaviors_embeddings)
        target_embeddings = self.func(target_embeddings)

        with tf.variable_scope(name_or_scope='temporal_interest_module'):
            input_mask = tf.sequence_mask(sequence_length, maxlen=user_behaviors_embeddings.shape.as_list()[1])
            query, key = self.merge_position_embedding(target_embeddings, user_behaviors_embeddings)
            attention_mask = create_attention_mask_from_input_mask(query, input_mask)
            attention_outputs = self.attention_layer(query, key, attention_mask)
            attention_outputs = tf.squeeze(attention_outputs, axis=1)

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], other_feature_ids[name]))

        dnn_inputs = tf.concat([attention_outputs, target_embeddings] + other_feature_embeddings, axis=-1)

        with tf.variable_scope(name_or_scope='mlps'):
            output = self.mlp_layer(dnn_inputs, is_training=is_training)

        output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())
        return tf.reshape(output, [-1])

