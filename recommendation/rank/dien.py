"""
论文：Deep Interest Evolution Network for Click-Through Rate Prediction

链接：https://arxiv.org/pdf/1809.03672.pdf

references：https://github.com/mouna99/dien
"""
from typing import List, Callable, Optional, Dict, Type
from functools import partial
# from tensorflow.python.ops.rnn import dynamic_rnn
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from ..utils.core import dnn_layer, dice
from ..utils.interaction import AttentionBase
from ..utils.type_declaration import DINField
from .din import BaseModelMiniBatchReg


class DIEN(BaseModelMiniBatchReg):
    def __init__(self,
                 fields: List[DINField],
                 mlp_hidden_units: List[int],
                 attention_gru: Type[AttentionBase],
                 gru_hidden_size: int,
                 mlp_activation: Callable = dice,
                 mlp_dropout: Optional[float] = 0.,
                 mlp_use_bn: Optional[bool] = True,
                 mlp_l2_reg: float = 0.,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = tf.nn.sigmoid,
                 mode: str = 'concat'):
        super().__init__(fields, mode)
        self.mlp_layer = partial(dnn_layer, hidden_units=mlp_hidden_units, activation=mlp_activation, use_bn=mlp_use_bn,
                                 dropout=mlp_dropout, l2_reg=mlp_l2_reg)
        self.attention_gru = attention_gru(gru_hidden_size, attention_hidden_units, attention_activation)
        self.auxiliary_obj = AuxiliaryLoss()

    def __call__(self,
                 user_behaviors_ids: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 target_ids: Dict[str, tf.Tensor],
                 negative_sequence_ids: Dict[str, tf.Tensor],
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
                                                              user_behaviors_ids[name],
                                                              negative_sequence_ids[name]], axis=-1))
            concat_item_embeddings.append(embeddings)
        concat_item_embeddings = self.func(concat_item_embeddings)  # [B, N+1, D']
        target_embedding = concat_item_embeddings[:, 0, :]  # [B, D']
        # # [B, N, D']
        user_behaviors_embedding, negative_sequence_embedding = tf.split(concat_item_embeddings[:, 1:, :], 2, axis=1)
        # negative_sequence_embedding = concat_item_embeddings[:, 1, :]  # [B, D']
        # user_behaviors_embedding = concat_item_embeddings[:, 2:, :]  # [B, N, D']

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(self.embedding_lookup(name, other_feature_ids[name]))
        other_feature_embeddings = tf.concat(other_feature_embeddings, axis=-1)

        with tf.variable_scope(name_or_scope='attention_gru'):
            interest_states, _, final_state = self.attention_gru(user_behaviors_embedding, target_embedding, sequence_length)

        dnn_inputs = tf.concat([final_state, target_embedding, other_feature_embeddings],
                               axis=-1)
        with tf.variable_scope(name_or_scope='mlp_layer'):
            output = self.mlp_layer(inputs=dnn_inputs, is_training=is_training)
            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                     # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                     kernel_initializer=tf.glorot_normal_initializer())
        with tf.variable_scope(name_or_scope='auxiliary_loss'):
            auxiliary_loss = self.auxiliary_obj(interest_states[:, :-1, :],
                                                user_behaviors_embedding[:, 1:, :], negative_sequence_embedding[:, 1:, :],
                                                sequence_length-1)

        return tf.reshape(output, [-1]), auxiliary_loss


class AuxiliaryLoss:
    def __call__(self, h_states, click_seq, noclick_seq, seq_len, stag=''):
        mask = tf.sequence_mask(seq_len, tf.shape(click_seq)[1])
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        mask = tf.cast(mask, click_prop_.dtype)
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat
