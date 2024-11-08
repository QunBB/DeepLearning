"""
Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling

CIKM'2022：https://arxiv.org/abs/2201.06814
"""
from typing import List, Dict
from functools import partial
import tensorflow as tf

from ..utils.transformer import attention_layer, create_attention_mask_from_input_mask


class M2M:
    def __init__(self,
                 num_experts: int,
                 num_meta_unit_layer: int,
                 num_residual_layer: int,
                 shared_meta_unit: bool,
                 num_attention_heads: int,
                 attention_head_size: int,
                 views_dim: int,
                 num_sequence: int,
                 max_len_sequence: int,
                 position_embedding_dim: int = 16,
                 l2_reg: float = 0.
                 ):
        """

        :param num_experts: MMoE中专家的数量
        :param num_meta_unit_layer: meta unit的层数
        :param num_residual_layer: meta tower模块的层数
        :param shared_meta_unit: 是否共享meta unit的参数
        :param num_attention_heads: transformer layer中的注意力头的数量
        :param attention_head_size: transformer layer中的注意力输出维度
        :param views_dim: expert view和task view的维度
        :param num_sequence: 多少种历史序列
        :param max_len_sequence: 最大的序列长度
        :param position_embedding_dim: transformer layer位置嵌入的维度
        :param l2_reg: 正则惩罚项
        """
        self.num_experts = num_experts
        self.num_meta_unit_layer = num_meta_unit_layer
        self.num_residual_layer = num_residual_layer
        self.l2_reg = l2_reg

        self.position_embeddings = []
        for i in range(num_sequence):
            self.position_embeddings.append(
                tf.get_variable(f'position_embedding_{i}', [1, max_len_sequence, position_embedding_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)),
            )

        self.mlp = partial(tf.layers.dense,
                           units=views_dim,
                           activation=tf.nn.leaky_relu,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                           kernel_initializer=tf.glorot_normal_initializer())

        self.attention_layer = partial(attention_layer,
                                       num_attention_heads=num_attention_heads,
                                       size_per_head=attention_head_size
                                       )

        self.shared_meta_unit = shared_meta_unit

    def meta_unit(self, input_tensor, scenario_views, num_layer, l2_reg):
        """Meta Unit"""
        input_shape = input_tensor.shape.as_list()
        dim = input_shape[-1]
        if len(input_shape) != 3:
            input_tensor = tf.expand_dims(input_tensor, axis=1)
        outputs = input_tensor
        for i in range(num_layer):
            w = tf.layers.dense(scenario_views, dim * dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                           kernel_initializer=tf.glorot_normal_initializer(), name=f"meta_unit_w_{i}", reuse=tf.AUTO_REUSE)
            b = tf.layers.dense(scenario_views, dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                           kernel_initializer=tf.glorot_normal_initializer(), name=f"meta_unit_b_{i}", reuse=tf.AUTO_REUSE)
            outputs = tf.matmul(outputs, tf.reshape(w, [-1, dim, dim])) + tf.expand_dims(b, axis=1)
            outputs = tf.nn.leaky_relu(outputs)

        return tf.squeeze(outputs)

    def residual_layer(self, input_tensor, scenario_views, num_layer, num_meta_unit_layer, l2_reg):
        """Meta Tower Module"""
        outputs = input_tensor
        for i in range(num_layer):
            outputs = self.meta_unit(outputs, scenario_views, num_meta_unit_layer, l2_reg) + outputs
            outputs = tf.nn.leaky_relu(outputs)

        return outputs

    def __call__(self,
                 multi_history_embeddings_list: List[tf.Tensor],
                 multi_history_len_list: List[tf.Tensor],
                 user_embeddings: tf.Tensor,
                 scenario_embeddings: tf.Tensor,
                 task_embeddings: Dict[str, tf.Tensor],
                 target_embeddings: tf.Tensor = None,
                 compute_logit: bool = False
                 ):
        """

        :param multi_history_embeddings_list: 历史序列embeddings列表，列表一个元素对应一种序列: bs * seq_len * dim
        :param multi_history_len_list: 序列长度列表，顺序必须与`multi_history_embeddings_list`相同
        :param user_embeddings: 广告主或用户特征embeddings: bs * dim
        :param scenario_embeddings: 场景特征embeddings: bs * dim
        :param task_embeddings: 任务特征embeddings: bs * dim
        :param target_embeddings: 适用于分类任务，target item的特征embeddings: bs * dim
        :param compute_logit: 适用于分类任务，是否计算概率
        :return:
        """
        # Scenario Knowledge Representation
        scenario_inputs = [scenario_embeddings, user_embeddings]
        if target_embeddings is not None:
            scenario_inputs.append(target_embeddings)
        scenario_views = self.mlp(tf.concat(scenario_inputs, axis=-1), name='scenario-mlp')

        # Transformer Layer
        multi_heads = []
        for idx, (input_tensor, length) in enumerate(zip(multi_history_embeddings_list, multi_history_len_list)):
            with tf.variable_scope(name_or_scope=f'multi-head-self-attention-{idx}'):
                input_mask = tf.sequence_mask(length, maxlen=input_tensor.shape.as_list()[1])
                attention_mask = create_attention_mask_from_input_mask(input_tensor, input_mask)
                input_tensor = tf.concat(
                    [input_tensor, tf.repeat(self.position_embeddings[idx], input_tensor.shape[0], axis=0)], axis=-1
                )
                transformer_outputs = self.attention_layer(input_tensor, input_tensor, attention_mask)

            # Transformer Layer for sequence embeddings aggregation
            with tf.variable_scope(name_or_scope=f'attention-aggregation-{idx}'):
                target_tensor = tf.expand_dims(scenario_views, axis=1)
                attention_mask = create_attention_mask_from_input_mask(target_tensor, input_mask)
                agg = self.attention_layer(target_tensor, transformer_outputs, attention_mask)
                multi_heads.append(tf.squeeze(agg, axis=1))

        # Expert View Representation
        expert_inputs = tf.concat(multi_heads + [user_embeddings], axis=-1)
        expert_views = []
        for i in range(self.num_experts):
            expert_views.append(self.mlp(expert_inputs, name=f'expert-mlp-{i}'))
        expert_views = tf.stack(expert_views, axis=1)

        task_outputs = {}
        for task in task_embeddings:
            # Task View Representation
            task_views = self.mlp(task_embeddings[task], name=f'task-mlp-{task}')

            # Meta Attention Module
            scope = 'meta-attention' if self.shared_meta_unit else f'meta-attention-{task}'
            with tf.variable_scope(name_or_scope=scope):
                meta = self.meta_unit(
                    tf.concat([expert_views, tf.repeat(tf.expand_dims(task_views, axis=1), self.num_experts, axis=1)], axis=-1),
                    scenario_views, self.num_meta_unit_layer, self.l2_reg)
                meta_attention_score = tf.layers.dense(inputs=meta, units=1, name=f'scalar-dense-{task}')
                meta_attention_output = tf.reduce_sum(expert_views * meta_attention_score, axis=1)

            # Meta Tower Module
            scope = 'meta-tower' if self.shared_meta_unit else f'meta-tower-{task}'
            with tf.variable_scope(name_or_scope=scope):
                meta_tower_output = self.residual_layer(meta_attention_output, scenario_views,
                                                        self.num_residual_layer, self.num_meta_unit_layer, self.l2_reg)

            # Prediction Layer
            prediction = tf.layers.dense(meta_tower_output, 1, kernel_initializer=tf.glorot_normal_initializer())
            prediction = tf.squeeze(prediction)
            if compute_logit:
                prediction = tf.nn.sigmoid(prediction)
            task_outputs[task] = prediction

        return task_outputs
