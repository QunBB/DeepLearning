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
from ..utils.rnn import dynamic_rnn
from .din import BaseModelMiniBatchReg, DINField, attention


class AttGRUBase:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DIEN(BaseModelMiniBatchReg):
    def __init__(self,
                 fields: List[DINField],
                 mlp_hidden_units: List[int],
                 attention_gru: Type[AttGRUBase],
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


class AIGRU(AttGRUBase):
    def __init__(self, hidden_size, attention_units, attention_activation):
        super().__init__(hidden_size, attention_units, attention_activation)
        self.hidden_size = hidden_size
        self.attention_units = attention_units
        self.attention_activation = attention_activation

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        with tf.name_scope('gru_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=item_his_emb,
                                         sequence_length=seq_len, dtype=tf.float32,
                                         scope="gru1")

        # Attention layer
        with tf.name_scope('attention_layer'):
            attention_score = attention(queries=target_item_emb, keys=rnn_outputs,
                                        keys_length=seq_len, queries_ffn=True,
                                        ffn_hidden_units=self.attention_units, ffn_activation=self.attention_activation,
                                        return_attention_score=True)
            att_outputs = rnn_outputs * tf.transpose(attention_score, [0, 2, 1])

        with tf.name_scope('gru_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(self.hidden_size), inputs=att_outputs,
                                                     sequence_length=seq_len, dtype=tf.float32,
                                                     scope="gru2")

        return rnn_outputs, rnn_outputs2, final_state2


class AGRU(AttGRUBase):
    def __init__(self, hidden_size, attention_units, attention_activation):
        super().__init__(hidden_size, attention_units, attention_activation)
        self.hidden_size = hidden_size
        self.attention_units = attention_units
        self.attention_activation = attention_activation

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        # RNN layer(-s)
        with tf.name_scope('gru_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=item_his_emb,
                                         sequence_length=seq_len, dtype=tf.float32,
                                         scope="gru1")

        # Attention layer
        with tf.name_scope('attention_layer'):
            attention_score = attention(queries=target_item_emb, keys=rnn_outputs, keys_length=seq_len, queries_ffn=True,
                                        ffn_hidden_units=self.attention_units, ffn_activation=self.attention_activation,
                                        return_attention_score=True)

        with tf.name_scope('gru_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(AttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                     att_scores=tf.transpose(attention_score, [0, 2, 1]),
                                                     sequence_length=seq_len, dtype=tf.float32,
                                                     scope="gru2")

        return rnn_outputs, rnn_outputs2, final_state2


class AUGRU(AttGRUBase):
    def __init__(self, hidden_size, attention_units, attention_activation):
        super().__init__(hidden_size, attention_units, attention_activation)
        self.hidden_size = hidden_size
        self.attention_units = attention_units
        self.attention_activation = attention_activation

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        # RNN layer(-s)
        with tf.name_scope('gru_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=item_his_emb,
                                         sequence_length=seq_len, dtype=tf.float32,
                                         scope="gru1")

        # Attention layer
        with tf.name_scope('attention_layer'):
            attention_score = attention(queries=target_item_emb, keys=rnn_outputs, keys_length=seq_len, queries_ffn=True,
                                        ffn_hidden_units=self.attention_units, ffn_activation=self.attention_activation,
                                        return_attention_score=True)

        with tf.name_scope('gru_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                     att_scores=tf.transpose(attention_score, [0, 2, 1]),
                                                     sequence_length=seq_len, dtype=tf.float32,
                                                     scope="gru2")

        return rnn_outputs, rnn_outputs2, final_state2


class AttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(AttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            gate_inputs = tf.layers.dense(
                array_ops.concat([inputs, state], 1),
                2 * self._num_units,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=bias_ones
            )

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        with vs.variable_scope("candidate"):
            candidate = tf.layers.dense(
                array_ops.concat([inputs, r_state], 1),
                self._num_units,
                bias_initializer=self._bias_initializer,
                kernel_initializer=self._kernel_initializer)
        c = self._activation(candidate)
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            gate_inputs = tf.layers.dense(
                array_ops.concat([inputs, state], 1),
                2 * self._num_units,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=bias_ones
            )

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        with vs.variable_scope("candidate"):
            candidate = tf.layers.dense(
                array_ops.concat([inputs, r_state], 1),
                self._num_units,
                bias_initializer=self._bias_initializer,
                kernel_initializer=self._kernel_initializer)
        c = self._activation(candidate)
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h
