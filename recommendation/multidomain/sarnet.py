"""
CIKM'2021：SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios

https://arxiv.org/pdf/2110.06475
"""
from functools import partial
from typing import List, Callable, Dict

import tensorflow as tf

from ..utils.interaction import attention
from ..utils.type_declaration import Field


class SARNet:
    def __init__(self,
                 fields: List[Field],
                 num_scenario: int,
                 num_scenario_experts: int,
                 num_shared_experts: int,
                 expert_inputs_dim: int,
                 fairness_coefficient_field_name: str = None,
                 l2_reg: float = 0.,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = tf.nn.sigmoid,
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param num_scenario: 场景数量
        :param num_scenario_experts: 场景特定专家数量
        :param num_shared_experts: 共享专家数量
        :param expert_inputs_dim: 混合专家层的输入维度
        :param fairness_coefficient_field_name: 公平系数对应的field名称
        :param l2_reg: 正则惩罚项
        :param attention_hidden_units: Attention参数
        :param attention_activation: Attention参数
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """
        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg)
                                                               )

        # Scenario-Specific Transform Layer
        self.scenario_beta = tf.get_variable('scenario_beta_table',
                                             shape=[num_scenario, expert_inputs_dim],
                                             initializer=tf.ones_initializer())
        self.scenario_gamma = tf.get_variable('scenario_gamma_table',
                                              shape=[num_scenario, expert_inputs_dim],
                                              initializer=tf.zeros_initializer())

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

        # Cross-Scenario Behavior Extract Layer
        self.item_attention = partial(attention, ffn_hidden_units=attention_hidden_units, ffn_activation=attention_activation, return_attention_score=True)
        self.scenario_attention = partial(attention, ffn_hidden_units=attention_hidden_units, ffn_activation=attention_activation, return_attention_score=True)

        # Mixture of Debias Experts
        self.fairness_coefficient_name = fairness_coefficient_field_name
        with tf.variable_scope(name_or_scope='scenario_specific_experts'):
            # 判断是否需要Bias Net
            net_list, dim_list = ['main_net'], [expert_inputs_dim]
            if fairness_coefficient_field_name and fairness_coefficient_field_name in self.embedding_table:
                net_list.append('bias_net')
                dim_list.append(self.embedding_table[fairness_coefficient_field_name].shape.as_list()[-1])
            self.scenario_experts_weight = {
                name: [
                    tf.get_variable(f'scenario_experts_{name}_w_table_{i}',
                                    shape=[num_scenario, dim],
                                    initializer=tf.glorot_normal_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
                                    )
                    for i in range(num_scenario_experts)
                ] for name, dim in zip(net_list, dim_list)
            }
            self.scenario_experts_bias = {
                name: [
                    tf.get_variable(f'scenario_experts_{name}_b_table_{i}',
                                    shape=[num_scenario, 1])
                    for i in range(num_scenario_experts)
                ] for name in net_list
            }

        self.num_scenario_experts = num_scenario_experts
        self.num_shared_experts = num_shared_experts

        self.l2_reg = l2_reg

    def expert_net(self, inputs, is_training, l2_reg, w=None, b=None):
        """单个专家层"""
        inputs = tf.layers.batch_normalization(inputs, training=is_training)
        if w is None:
            outputs = tf.layers.dense(inputs, 1,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                      kernel_initializer=tf.glorot_normal_initializer())
        else:
            outputs = tf.matmul(tf.expand_dims(inputs, axis=1), tf.expand_dims(w, axis=-1))
            outputs = tf.reshape(outputs, [-1, 1]) + b
        return outputs

    def debias_experts(self, inputs, scenario_ids, num_experts, is_training, experts_weight=None, experts_bias=None, fairness_coefficient_embedding=None):
        """Debias Expert Net"""
        experts_output = []
        for i in range(num_experts):
            main_net_outputs = self.expert_net(inputs, is_training, l2_reg=self.l2_reg,
                                               w=tf.nn.embedding_lookup(experts_weight['main_net'][i], scenario_ids) if experts_weight is not None else None,
                                               b=tf.nn.embedding_lookup(experts_bias['main_net'][i], scenario_ids) if experts_weight is not None else None)
            if is_training and fairness_coefficient_embedding is not None:
                bias_net_outputs = self.expert_net(fairness_coefficient_embedding, is_training, l2_reg=self.l2_reg,
                                                   w=tf.nn.embedding_lookup(experts_weight['bias_net'][i], scenario_ids) if experts_weight is not None else None,
                                                   b=tf.nn.embedding_lookup(experts_bias['bias_net'][i], scenario_ids) if experts_bias is not None else None)
                experts_output.append(main_net_outputs + bias_net_outputs)
            else:
                experts_output.append(main_net_outputs)

        return experts_output

    def __call__(self,
                 user_behaviors_items_sequence: Dict[str, tf.Tensor],
                 user_behaviors_scenario_context_sequence: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 target_ids: Dict[str, tf.Tensor],
                 scenario_context: Dict[str, tf.Tensor],
                 other_feature_ids: Dict[str, tf.Tensor],
                 scenario_ids: tf.Tensor,
                 fairness_coefficient: tf.Tensor = None,
                 is_training: bool = True
                 ):
        """

        :param user_behaviors_items_sequence: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id
        :param user_behaviors_scenario_context_sequence: 用户行为序列对应的场景上下文ID [B, N], 支持多种属性组合，如scenario_id+shop_id+scenario_type
        :param sequence_length: 用户行为序列长度 [B]
        :param target_ids: 候选item特征ID [B]
        :param scenario_context: 场景上下文特征ID [B]
        :param scenario_ids: 每个样本的场景ID [B]
        :param other_feature_ids: 其他特征，如用户特征及上下文特征
        :param fairness_coefficient: 每个样本的公平系数
        :param is_training:
        :return:
        """
        # 用户行为历史items embedding
        user_behaviors_item_embeddings = []
        target_embeddings = []
        for name in user_behaviors_items_sequence:
            user_behaviors_item_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], user_behaviors_items_sequence[name]))
            target_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], target_ids[name]))
        user_behaviors_item_embeddings = self.func(user_behaviors_item_embeddings)
        target_embeddings = self.func(target_embeddings)

        # 用户行为历史对应的scenario context embedding
        user_behaviors_scenario_context_embeddings = []
        scenario_context_embeddings = []
        for name in user_behaviors_scenario_context_sequence:
            user_behaviors_scenario_context_embeddings.append(
                tf.nn.embedding_lookup(self.embedding_table[name], user_behaviors_scenario_context_sequence[name]))
            scenario_context_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], scenario_context[name]))
        user_behaviors_scenario_context_embeddings = self.func(user_behaviors_scenario_context_embeddings)
        scenario_context_embeddings = self.func(scenario_context_embeddings)

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], other_feature_ids[name]))
        other_feature_embeddings = tf.concat(other_feature_embeddings, axis=-1)

        # Cross-Scenario Behavior Extract Layer
        # item序列attention和场景上下文序列attention
        with tf.variable_scope(name_or_scope='item_attention_layer'):
            item_attention_score = self.item_attention(keys=user_behaviors_item_embeddings,
                                                       queries=target_embeddings,
                                                       keys_length=sequence_length)
        with tf.variable_scope(name_or_scope='scenario_attention_layer'):
            scenario_attention_score = self.item_attention(keys=user_behaviors_scenario_context_embeddings,
                                                           queries=scenario_context_embeddings,
                                                           keys_length=sequence_length)
        interest_transfer_vector = tf.matmul(item_attention_score * scenario_attention_score, user_behaviors_item_embeddings)

        mixture_experts_input = tf.concat([tf.squeeze(interest_transfer_vector), target_embeddings, scenario_context_embeddings, other_feature_embeddings], axis=-1)

        # Scenario-Specific Transform Layer
        mixture_experts_input = mixture_experts_input * tf.nn.embedding_lookup(self.scenario_beta, scenario_ids) + tf.nn.embedding_lookup(self.scenario_gamma, scenario_ids)

        # Mixture of Debias Experts
        fairness_coefficient_embedding = None
        if is_training and fairness_coefficient is not None:
            fairness_coefficient_embedding = tf.nn.embedding_lookup(self.embedding_table[self.fairness_coefficient_name], scenario_ids)
            fairness_coefficient_embedding = fairness_coefficient_embedding * tf.reshape(fairness_coefficient, [-1, 1])

        scenario_experts = self.debias_experts(mixture_experts_input, scenario_ids, self.num_scenario_experts, is_training,
                                               experts_weight=self.scenario_experts_weight,
                                               experts_bias=self.scenario_experts_bias,
                                               fairness_coefficient_embedding=fairness_coefficient_embedding)
        shared_experts = self.debias_experts(mixture_experts_input, scenario_ids, self.num_shared_experts, is_training,
                                             fairness_coefficient_embedding=fairness_coefficient_embedding)

        # Multi-Gate Network
        gates = tf.layers.dense(mixture_experts_input, self.num_scenario_experts + self.num_shared_experts,
                                kernel_initializer=tf.glorot_normal_initializer())
        gates = tf.nn.softmax(gates, axis=-1)

        logit = tf.reduce_sum(tf.concat(scenario_experts + shared_experts, axis=-1) * gates, axis=-1)

        return tf.nn.sigmoid(logit)
