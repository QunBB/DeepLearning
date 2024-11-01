from typing import List, Callable, Optional, Dict, Type, Union
from functools import partial

import tensorflow as tf

from ..utils.type_declaration import Field
from ..utils.interaction import AttentionBase, Attention


class GateNU:
    def __init__(self,
                 hidden_units,
                 gamma=2.,
                 l2_reg=0.):
        assert len(hidden_units) == 2
        self.hidden_units = hidden_units
        self.gamma = gamma
        self.l2_reg = l2_reg

    def __call__(self, inputs):
        output = tf.layers.dense(inputs, self.hidden_units[0], activation="relu",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
        output = tf.layers.dense(output, self.hidden_units[1], activation="sigmoid",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
        return self.gamma * output


class EPNet:
    def __init__(self,
                 hidden_units,
                 l2_reg=0.):

        self.gate_nu = GateNU(hidden_units=hidden_units, l2_reg=l2_reg)

    def __call__(self, domain, emb):
        return self.gate_nu(tf.concat([domain, tf.stop_gradient(emb)], axis=-1)) * emb


class PPNet:
    def __init__(self,
                 multiples,
                 hidden_units,
                 activation,
                 l2_reg=0.,
                 **kwargs):
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg
        self.activation = activation

        self.multiples = multiples

        self.gate_nu = [GateNU([i*self.multiples, i*self.multiples], l2_reg=self.l2_reg) for i in self.hidden_units]

    def __call__(self, inputs, persona):
        gate_list = []
        for i in range(len(self.hidden_units)):
            gate = self.gate_nu[i](tf.concat([persona, tf.stop_gradient(inputs)], axis=-1))
            gate = tf.split(gate, self.multiples, axis=1)
            gate_list.append(gate)

        output_list = []

        for n in range(self.multiples):
            output = inputs

            for i in range(len(self.hidden_units)):
                fc = tf.layers.dense(output, self.hidden_units[i], activation=self.activation,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

                output = gate_list[i][n] * fc

            output_list.append(output)

        return output_list


class PEPNet:
    def __init__(self,
                 fields: List[Field],
                 num_tasks: int,
                 dnn_hidden_units: List[int] = [100, 64],
                 dnn_activation: Union[str, Callable] = "relu",
                 dnn_l2_reg: float = 0.,
                 attention_agg: Type[AttentionBase] = Attention,
                 gru_hidden_size: int = 1,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = tf.nn.sigmoid,
                 mode: str = "concat"):
        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(
                                                                   field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(
                                                                   field.l2_reg)
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

        self.num_tasks = num_tasks

        self.epnet = partial(EPNet, l2_reg=dnn_l2_reg)
        self.ppnet = PPNet(num_tasks, dnn_hidden_units, dnn_activation, dnn_l2_reg)

        with tf.variable_scope(name_or_scope='attention_layer'):
            self.attention_agg = attention_agg(gru_hidden_size, attention_hidden_units, attention_activation)

    def embedding(self, inputs_dict):
        result = []
        for name in inputs_dict:
            result.append(
                tf.nn.embedding_lookup(self.embedding_table[name], inputs_dict[name])
            )
        return self.func(result)

    def predict_layer(self, inputs):
        output = tf.layers.dense(inputs, 1, activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.glorot_normal_initializer())
        return tf.reshape(output, [-1])

    def __call__(self,
                 user_behaviors_ids: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 user_ids: Dict[str, tf.Tensor],
                 item_ids: Dict[str, tf.Tensor],
                 other_feature_ids: Dict[str, tf.Tensor],
                 domain_ids: Dict[str, Dict[str, tf.Tensor]],
                 ) -> Dict[str, List[tf.Tensor]]:
        """

        :param user_behaviors_ids: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id
        :param sequence_length: 用户行为序列长度 [B]
        :param user_ids: 用户个性化特征
        :param item_ids: 候选items个性化特征
        :param other_feature_ids: 其他特征，如用户特征及上下文特征
        :param domain_ids: 每个场景的所有特征，key为场景名称，value如上user_ids和item_ids等
        :return: 每个场景的所有task预估列表
        """
        user_behaviors_embeddings = self.embedding(user_behaviors_ids)
        user_embeddings = self.embedding(user_ids)
        item_embeddings = self.embedding(item_ids)

        domain_embeddings = {}
        for domain in domain_ids:
            domain_embeddings[domain] = self.embedding(domain_ids[domain])

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], other_feature_ids[name]))
        other_feature_embeddings = tf.concat(other_feature_embeddings, axis=-1)

        with tf.variable_scope(name_or_scope='attention_layer'):
            att_outputs = self.attention_agg(user_behaviors_embeddings, item_embeddings, sequence_length)
            if isinstance(att_outputs, (list, tuple)):
                att_outputs = att_outputs[-1]

        inputs = tf.concat([att_outputs, other_feature_embeddings], axis=-1)
        inputs_dim = inputs.shape.as_list()[-1]
        epnet = self.epnet(hidden_units=[inputs_dim, inputs_dim])

        output_dict = {}
        # compute each domain's prediction
        for domain in domain_embeddings:
            ep_emb = epnet(domain_embeddings[domain], inputs)

            pp_outputs = self.ppnet(ep_emb, tf.concat([user_embeddings, item_embeddings], axis=-1))

            # compute each task's prediction in special domain
            task_outputs = []
            for i in range(self.num_tasks):
                task_outputs.append(self.predict_layer(pp_outputs[i]))

            output_dict[domain] = task_outputs

        return output_dict
