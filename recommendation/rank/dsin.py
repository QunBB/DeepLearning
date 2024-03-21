import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from functools import partial
from typing import Dict, List, Callable

from ..utils.transformer import transformer_model, create_attention_mask_from_input_mask
from ..utils.type_declaration import Field
from ..utils.core import dnn_layer


class Embedding:
    def __init__(self,
                 fields: List[Field],
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """
        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg) if field.l2_reg else None
                                                               )

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

    def __call__(self, inputs_dict: dict):
        """
        输入转化为embeddings并进行聚合pooling
        :param inputs_dict:
        :return:
        """
        embeddings = []
        for name in inputs_dict:
            embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], inputs_dict[name]))

        return self.func(embeddings)


class BiasEncoding:
    def __init__(self, emb_size, num_sessions, seq_length_max, std=1e-4):
        # bias vector of the session
        self.sess_bias_emb = tf.get_variable('sess_bias_emb', [num_sessions, 1, 1],
                                             initializer=tf.truncated_normal_initializer(stddev=std))
        # bias vector of the position in the session
        self.seq_bias_emb = tf.get_variable('seq_bias_emb', [1, seq_length_max, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=std))
        #  bias vector of the unit position in the behavior embedding
        self.emb_bias_emb = tf.get_variable('emb_bias_emb', [1, 1, emb_size],
                                            initializer=tf.truncated_normal_initializer(stddev=std))

    def __call__(self, inputs, idx=None):
        if isinstance(inputs, list):
            output = []
            for i, tensor in enumerate(inputs):
                output.append(tensor + self.emb_bias_emb + self.seq_bias_emb + self.sess_bias_emb[i])
        else:
            output = inputs + self.emb_bias_emb + self.seq_bias_emb + self.sess_bias_emb[idx:(idx+1)]

        return output


class BiLSTM:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        # 前向 LSTM
        self.lstm_fw_cell = LSTMCell(hidden_size)
        # 反向 LSTM
        self.lstm_bw_cell = LSTMCell(hidden_size)

    def __call__(self, inputs, sequence_length=None):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell,
                                                     inputs, sequence_length=sequence_length, dtype=inputs.dtype)
        return outputs[0] + outputs[1]


class DSIN:
    def __init__(self,
                 fields: List[Field],
                 item_embedding_size: int,
                 num_sessions: int,
                 seq_length_max: int,
                 att_embedding_size: int,
                 lstm_hidden_size: int,
                 mlp_hidden_units: List[int],
                 num_attention_heads: int,
                 intermediate_size: int,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_pro: float = 0.1,
                 initializer_range: float = 0.02,
                 mlp_activation: Callable = tf.nn.relu,
                 mlp_use_bn: bool = True,
                 mlp_dropout: float = 0.,
                 mlp_l2_reg: float = 0.,
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param item_embedding_size: item的embedding size
        :param num_sessions: sessions的数量 K
        :param seq_length_max: 每个session中的行为序列最大长度 T
        :param att_embedding_size: 自注意力的输出size
        :param lstm_hidden_size: Bi-LSTM的隐状态size
        :param mlp_hidden_units: mlp隐藏层size列表
        :param mlp_activation: mlp隐藏层的激活函数
        :param num_attention_heads: 自注意力头的个数
        :param intermediate_size: 自注意力中的中间层size
        :param hidden_dropout_prob: 自注意力的隐藏层dropout
        :param attention_probs_dropout_pro: 自注意力的attention层dropout
        :param initializer_range: 自注意力的参数初始化方差
        :param mlp_use_bn: mlp层是否使用bn
        :param mlp_dropout: mlp层的dropout
        :param mlp_l2_reg: mlp层的参数正则惩罚
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """
        self.embedding_layer = Embedding(fields, mode)

        self.bias_encoding = BiasEncoding(item_embedding_size, num_sessions, seq_length_max)

        self.bi_lstm = BiLSTM(lstm_hidden_size)

        self.transformer_model = partial(transformer_model,
                                         hidden_size=att_embedding_size,
                                         num_hidden_layers=1,
                                         num_attention_heads=num_attention_heads,
                                         intermediate_size=intermediate_size,
                                         hidden_dropout_prob=hidden_dropout_prob,
                                         attention_probs_dropout_prob=attention_probs_dropout_pro,
                                         initializer_range=initializer_range,
                                         )

        self.mlp_layer = partial(dnn_layer, hidden_units=mlp_hidden_units, activation=mlp_activation, use_bn=mlp_use_bn,
                                 dropout=mlp_dropout, l2_reg=mlp_l2_reg)

    def __call__(self,
                 user_session_behaviors_id: List[Dict[str, tf.Tensor]],
                 seq_length_list: List[tf.Tensor],
                 session_length: tf.Tensor,
                 item_profile_ids: Dict[str, tf.Tensor],
                 user_profile_ids: Dict[str, tf.Tensor],
                 is_training: bool = True
                 ):
        """

        :param user_session_behaviors_id: 用户每个session的行为序列ID [Bs, T, N]*K, 支持多种属性组合，如goods_id+shop_id+cate_id
        :param seq_length_list: 每个session的行为序列长度 [Bs]*K
        :param session_length: sessions的个数 [Bs] $\in \{0,1,...,T\}$
        :param item_profile_ids: target item ID [Bs]
        :param user_profile_ids: 用户属性 [Bs]
        :param is_training:
        :return:
        """
        target_item_emb = self.embedding_layer(item_profile_ids)
        user_profile_emb = self.embedding_layer(user_profile_ids)

        # 兴趣提取层 Session Interest Extractor Layer
        with tf.variable_scope(name_or_scope='session_interest_extractor_layer'):
            session_interests_emb = []
            for idx, (user_behaviors_id, seq_length) in enumerate(zip(user_session_behaviors_id, seq_length_list)):
                emb = self.embedding_layer(user_behaviors_id)
                session_interests_emb.append(session_interest_extract(self.bias_encoding(emb, idx=idx), seq_length,
                                                                      self.transformer_model, scope='session'))
            session_interests_emb = tf.stack(session_interests_emb, axis=1)

        # 兴趣交互层 Session Interest Interacting Layer
        with tf.variable_scope(name_or_scope='session_interest_interacting_layer'):
            session_lstm_outputs = self.bi_lstm(session_interests_emb, session_length)

        # 兴趣激活层 Session Interest Activating Layer
        with tf.variable_scope(name_or_scope='session_interest_activating_layer'):
            session_mask = tf.sequence_mask(session_length,
                                            maxlen=session_interests_emb.shape.as_list()[1],
                                            dtype=session_interests_emb.dtype)
            interests_attention = session_interest_activating(session_interests_emb, target_item_emb, session_mask)
            lstm_attention = session_interest_activating(session_lstm_outputs, target_item_emb, session_mask)

        # MLP
        dnn_inputs = tf.concat([user_profile_emb, target_item_emb, interests_attention, lstm_attention], axis=-1)
        with tf.variable_scope(name_or_scope='mlp_layer'):
            output = self.mlp_layer(inputs=dnn_inputs, is_training=is_training)
            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.glorot_normal_initializer())

        return tf.reshape(output, [-1])


def session_interest_extract(input_tensor, seq_length, model, scope=''):
    input_mask = tf.sequence_mask(seq_length, maxlen=input_tensor.shape.as_list()[1])
    if input_mask is not None:
        attention_mask = create_attention_mask_from_input_mask(
            input_tensor, input_mask)
    else:
        attention_mask = None

    # 所有sessions的自注意力参数是共享的
    with tf.variable_scope(f'transformer_{scope}', reuse=tf.AUTO_REUSE):
        attention_output = model(input_tensor, attention_mask)

    input_mask = tf.cast(input_mask, input_tensor.dtype)
    session_interest = tf.reduce_sum(attention_output * tf.expand_dims(input_mask, axis=-1), axis=1) / tf.reduce_sum(input_mask, axis=1, keepdims=True)

    return session_interest


def session_interest_activating(score, target, mask):
    # [bs, seq_len, 1]
    iwx = tf.matmul(
        tf.layers.dense(score, target.shape.as_list()[-1],
                        kernel_initializer=tf.glorot_normal_initializer(),
                        use_bias=False),
        tf.expand_dims(target, axis=-1)
    )
    # [bs, seq_len]
    iwx = tf.layers.flatten(iwx)
    # 让mask的位置等1，因为exp(1)=0
    iwx_with_mask = iwx * mask + (1 - mask)

    weights = tf.nn.softmax(iwx_with_mask, axis=1)

    outputs = tf.reduce_sum(
        score * tf.expand_dims(weights, axis=-1),
        axis=1)
    return outputs
