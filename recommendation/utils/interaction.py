import itertools
from collections import OrderedDict
from typing import List, Union
from typing import Dict as OrderedDictType

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from .type_declaration import Field, LinearTerms
from .core import dnn_layer, dice, prelu
from .rnn import dynamic_rnn


class LinearEmbedding:
    """
    embedding layer + linear
    """
    def __init__(self,
                 fields_list: List[Field],
                 linear_type: LinearTerms = LinearTerms.LW):
        self.embeddings_table = {}
        self.weights_table = {}

        for field in fields_list:
            # embeddings 隐向量
            self.embeddings_table[field.name] = tf.get_variable('emb_' + field.name,
                                                                shape=[field.vocabulary_size, field.dim],
                                                                regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))

            # 线性项权重
            if linear_type == LinearTerms.LW:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[field.vocabulary_size],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))
            elif linear_type == LinearTerms.FeLV:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name,
                                                                 shape=[field.vocabulary_size, field.dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))
            else:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[1, field.dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))

    def _get_linear_logit(self,
                          w: tf.Tensor,
                          i: Union[int, tf.Tensor] = 0,
                          x: Union[int, tf.Tensor] = 1,
                          v: Union[int, tf.Tensor] = 1
                          ):
        """线性项计算"""
        shape = w.shape.as_list()

        if len(shape) == 1:  # LM
            return tf.gather(w, i) * x
        elif len(shape) == 2 and shape[0] == 1:  # FiLV
            return tf.reduce_sum(w * v, axis=1) * x
        elif len(shape) == 2 and shape[0] > 1:  # FeLV
            return tf.reduce_sum(tf.gather(w, i) * v, axis=1) * x
        else:
            raise ValueError

    def __call__(self,
                 sparse_id_inputs_dict: OrderedDictType[str, tf.Tensor] = {},
                 dense_value_inputs_dict: OrderedDictType[str, tf.Tensor] = {},
                 as_dict: bool = False):
        linear_logit = []
        embeddings = OrderedDict()

        for name, x in sparse_id_inputs_dict.items():
            v = tf.nn.embedding_lookup(self.embeddings_table[name], x)
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], i=x, v=v))
            embeddings[name] = v

        for name, x in dense_value_inputs_dict.items():
            v = tf.reshape(self.embeddings_table[name][0], [1, -1])
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], x=x, v=v))
            embeddings[name] = v * tf.reshape(x, [-1, 1])

        linear_logit = tf.add_n(linear_logit)

        if not as_dict:
            embeddings = list(embeddings.values())

        return embeddings, linear_logit


class BiLinear:
    def __init__(self,
                 output_size: int,
                 bilinear_type: str,
                 equal_dim: bool = True,
                 bilinear_plus: bool = False,
                 ):
        """
        双线性特征交互层，支持不同field embeddings的size不等
        :param output_size: 输出的size
        :param bilinear_type: ['all', 'each', 'interaction']，支持其中一种
        :param equal_dim: 所有field embeddings的size是否相同
        :param bilinear_plus: 是否使用bi-linear+
        """
        self.bilinear_type = bilinear_type
        self.output_size = output_size

        if bilinear_type not in ['all', 'each', 'interaction']:
            raise NotImplementedError("bilinear_type only support: ['all', 'each', 'interaction']")

        # 当所有field embeddings的size不等时，bilinear_type只能为'interaction'
        if not equal_dim:
            self.bilinear_type = 'interaction'

        if bilinear_plus:
            self.func = self._full_interaction
        else:
            self.func = tf.multiply

    def __call__(self, embeddings_inputs: List[tf.Tensor]):
        field_size = len(embeddings_inputs)

        # field embeddings的size
        _dim = embeddings_inputs[0].shape.as_list()[-1]

        # bi-linear+: p的维度为[bs, f*(f-1)/2]
        # bi-linear:
        # 当equal_dim=True时，p的维度为[bs, f*(f-1)/2*k]，k为embeddings的size
        # 当equal_dim=False时，p的维度为[bs, (k_2+k_3+...+k_f)+...+(k_i+k_{i+1}+...+k_f)+...+k_f]，k_i为第i个field的embedding的size
        if self.bilinear_type == 'all':
            v_dot = [tf.layers.dense(v_i, _dim,
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     name='bilinear', reuse=tf.AUTO_REUSE)
                     for v_i in embeddings_inputs[:-1]]
            p = [self.func(v_dot[i], embeddings_inputs[j]) for i, j in itertools.combinations(range(field_size), 2)]
        elif self.bilinear_type == 'each':
            v_dot = [tf.layers.dense(v_i, _dim,
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     name=f'bilinear_{i}', reuse=tf.AUTO_REUSE)
                     for i, v_i in enumerate(embeddings_inputs[:-1])]
            p = [self.func(v_dot[i], embeddings_inputs[j])
                 for i, j in itertools.combinations(range(field_size), 2)]
        else:  # interaction
            p = [self.func(tf.layers.dense(embeddings_inputs[i], embeddings_inputs[j].shape.as_list()[-1],
                                           kernel_initializer=tf.glorot_normal_initializer(),
                                           name=f'bilinear_{i}_{j}', reuse=tf.AUTO_REUSE), embeddings_inputs[j])
                 for i, j in itertools.combinations(range(field_size), 2)]

        output = tf.layers.dense(tf.concat(p, axis=-1), self.output_size,
                                 kernel_initializer=tf.glorot_normal_initializer())
        return output

    def _full_interaction(self, v_i, v_j):
        # [bs, 1, dim] x [bs, dim, 1] = [bs, 1]
        interaction = tf.matmul(tf.expand_dims(v_i, axis=1), tf.expand_dims(v_j, axis=-1))
        return tf.reshape(interaction, [-1, 1])


class SENet:
    """
    SENet+ Layer，支持不同field embeddings的size不等
    """

    def __init__(self,
                 reduction_ratio: int,
                 num_groups: int):
        self.reduction_ratio = reduction_ratio
        self.num_groups = num_groups

    def __call__(self, embeddings_list: List[tf.Tensor]):
        """

        :param embeddings_list: [embedding_1,...,embedding_i,...,embedding_f]，f为field的数目，embedding_i is [bs, dim]
        :return:
        """
        for emb in embeddings_list:
            assert len(emb.shape.as_list()) == 2, 'field embeddings must be rank 2 tensors'

        field_size = len(embeddings_list)
        feature_size_list = [emb.shape.as_list()[-1] for emb in embeddings_list]

        # Squeeze
        group_embeddings_list = [tf.reshape(emb, [-1, self.num_groups, tf.shape(emb)[-1] // self.num_groups]) for emb in
                                 embeddings_list]
        Z = [tf.reduce_mean(emb, axis=-1) for emb in group_embeddings_list] + [tf.reduce_max(emb, axis=-1) for emb in
                                                                               group_embeddings_list]
        Z = tf.concat(Z, axis=1)  # [bs, field_size * num_groups * 2]

        # Excitation
        reduction_size = max(1, field_size * self.num_groups * 2 // self.reduction_ratio)

        A_1 = tf.layers.dense(Z, reduction_size,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              activation=tf.nn.relu,
                              name='W_1')
        A_2 = tf.layers.dense(A_1, sum(feature_size_list),
                              kernel_initializer=tf.glorot_normal_initializer(),
                              activation=tf.nn.relu,
                              name='W_2')

        # Re-weight & Fuse
        senet_plus_embeddings = [emb * w + emb for emb, w in zip(embeddings_list, tf.split(A_2, feature_size_list, axis=1))]

        # Layer Normalization
        senet_plus_output = [tf.contrib.layers.layer_norm(
            inputs=x, begin_norm_axis=-1, begin_params_axis=-1, scope=f'LN-{i}') for i, x in enumerate(senet_plus_embeddings)]

        return tf.concat(senet_plus_output, axis=-1)


def attention(queries, keys, keys_length,
              ffn_hidden_units=[80, 40], ffn_activation=dice,
              queries_ffn=False, queries_activation=prelu,
              return_attention_score=False):
    """

    :param queries: [B, H]
    :param keys: [B, T, X]
    :param keys_length: [B]
    :param queries_ffn: 是否对queries进行一次ffn
    :param queries_activation: queries ffn的激活函数
    :param ffn_hidden_units: 隐藏层的维度大小
    :param ffn_activation: 隐藏层的激活函数
    :param return_attention_score: 是否返回注意力得分
    :return: attention_score=[B, 1, T] or attention_outputs=[B, H]
    """
    if queries_ffn:
        queries = tf.layers.dense(queries, keys.get_shape().as_list()[-1], name='queries_ffn')
        queries = queries_activation(queries)
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
    attention_score = tf.nn.softmax(outputs)  # [B, 1, T]

    if return_attention_score:
        return attention_score

    # Weighted sum
    attention_outputs = tf.matmul(attention_score, keys)  # [B, 1, H]

    return tf.squeeze(attention_outputs)


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


class AttentionBase:
    def __init__(self, hidden_size, attention_units, attention_activation):
        self.hidden_size = hidden_size
        self.attention_units = attention_units
        self.attention_activation = attention_activation

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        raise NotImplementedError


class Attention(AttentionBase):

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        return attention(target_item_emb, item_his_emb, seq_len,
                         ffn_hidden_units=self.attention_units, ffn_activation=self.attention_activation)


class AIGRU(AttentionBase):

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


class AGRU(AttentionBase):

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        # RNN layer(-s)
        with tf.name_scope('gru_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=item_his_emb,
                                         sequence_length=seq_len, dtype=tf.float32,
                                         scope="gru1")

        # Attention layer
        with tf.name_scope('attention_layer'):
            attention_score = attention(queries=target_item_emb, keys=rnn_outputs, keys_length=seq_len,
                                        queries_ffn=True,
                                        ffn_hidden_units=self.attention_units, ffn_activation=self.attention_activation,
                                        return_attention_score=True)

        with tf.name_scope('gru_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(AttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                     att_scores=tf.transpose(attention_score, [0, 2, 1]),
                                                     sequence_length=seq_len, dtype=tf.float32,
                                                     scope="gru2")

        return rnn_outputs, rnn_outputs2, final_state2


class AUGRU(AttentionBase):

    def __call__(self, item_his_emb, target_item_emb, seq_len):
        # RNN layer(-s)
        with tf.name_scope('gru_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=item_his_emb,
                                         sequence_length=seq_len, dtype=tf.float32,
                                         scope="gru1")

        # Attention layer
        with tf.name_scope('attention_layer'):
            attention_score = attention(queries=target_item_emb, keys=rnn_outputs, keys_length=seq_len,
                                        queries_ffn=True,
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
