"""
AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System

DASFAA'2025：https://arxiv.org/abs/2501.15816
"""
from collections import OrderedDict
from functools import partial
from typing import Optional, Callable, List, Dict, Any
from typing import Dict as OrderedDictType

import tensorflow as tf

from .interaction_expert import Expert
from ..utils.type_declaration import Field
from ..utils.core import dnn_layer
from ..utils.interaction import MaskEmbedding, StateAwareAdapter


class AdaF2M2:
    def __init__(
            self,
            fields_list: List[Field],
            state_id_fields: List[Field],
            state_non_id_fields: List[Field],
            num_sample: int,
            interaction: Expert,
            hidden_units: List[int],
            interaction_params: Optional[Dict[str, Any]] = None,
            min_probability: float = 0.1,
            max_probability: float = 0.5,
            embed_dense: bool = True,
            state_norm_clip_min: Optional[float] = None,
            state_norm_clip_max: Optional[float] = None,
            dropout: float = 0.,
            l2_reg: float = 0.,
            activation: Optional[Callable] = tf.nn.relu,
            use_bn: bool = True,
    ):
        """Adaptive Feature Modeling with Feature Mask

        :param fields_list: 特征列表
        :param state_non_id_fields: 状态信号的非ID特征列表
        :param state_id_fields: 状态信号的ID特征列表，会拼接原embedding和其norm
        :param state_norm_clip_min: embedding norm的下界
        :param state_norm_clip_max: embedding norm的上界
        :param embed_dense: 是否对状态信号输入进行embedding映射
        :param num_sample: 特征掩码中的增强样本数量
        :param min_probability: 特征掩码的概率区间的下界
        :param max_probability: 特征掩码的概率区间的上界
        :param interaction: 特征交互专家
        :param interaction_params: 特征交互专家的初始化参数
        :param hidden_units: DNN的每一层隐藏层大小
        :param dropout:
        :param l2_reg: 正则惩罚
        :param activation: 激活函数
        :param use_bn: 是否使用batch_normalization
        """
        self.num_sample = num_sample
        self.min_probability = min_probability
        self.max_probability = max_probability

        self.main_features = [f.name for f in fields_list]
        self.feature_mask = MaskEmbedding(fields_list, num_sample,
                                          min_probability=min_probability, max_probability=max_probability)
        self.state_aware_adapter = StateAwareAdapter(len(fields_list), state_id_fields, state_non_id_fields,
                                                     embed_dense=embed_dense,
                                                     embeddings_table=self.feature_mask.embeddings_table,
                                                     norm_clip_min=state_norm_clip_min,
                                                     norm_clip_max=state_norm_clip_max)
        self.state_id_names = [f.name for f in state_id_fields]
        self.state_non_id_names = [f.name for f in state_non_id_fields]

        self.interaction_layer = interaction.init_layer(**(interaction_params or {}))
        self.dnn_layer = partial(dnn_layer, hidden_units=hidden_units, activation=activation,
                                 dropout=dropout, use_bn=use_bn, l2_reg=l2_reg)

    def __call__(
        self,
        sparse_id_inputs_dict: OrderedDictType[str, tf.Tensor],
        dense_value_inputs_dict: OrderedDictType[str, tf.Tensor] = {},
        is_training: bool = True
    ):
        embeddings, augment_embeddings = self.feature_mask(
            sparse_id_inputs_dict=OrderedDict([(k, v) for k, v in sparse_id_inputs_dict.items() if k in self.main_features]),
            dense_value_inputs_dict=OrderedDict([(k, v) for k, v in dense_value_inputs_dict.items() if k in self.main_features])
        )

        # Feature Mask
        augmented_interactions = self.interaction_layer(augment_embeddings)
        augmented_outputs = self.dnn_layer(augmented_interactions, is_training=is_training)

        # Adaptive Feature Modeling
        adaptive_weights = self.state_aware_adapter(
            sparse_id_inputs_dict=OrderedDict([(k, v) for k, v in sparse_id_inputs_dict.items() if k in self.state_id_names]),
            dense_value_inputs_dict=OrderedDict([(k, v) for k, v in dense_value_inputs_dict.items() if k in self.state_non_id_names])
        )
        adaptive_embeddings = [emb * adaptive_weights[:, i:(i+1)] for i, emb in enumerate(embeddings)]
        main_interactions = self.interaction_layer(adaptive_embeddings)
        main_outputs = self.dnn_layer(main_interactions, is_training=is_training)

        # Predict
        main_outputs = tf.layers.dense(main_outputs, 1, activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.glorot_normal_initializer())
        augmented_outputs = tf.layers.dense(augmented_outputs, 1, activation=tf.nn.sigmoid,
                                       kernel_initializer=tf.glorot_normal_initializer())

        # main_outputs: [batch_size], augmented_outputs: [batch_size, num_sample]
        return tf.reshape(main_outputs, [-1]), tf.reshape(augmented_outputs, [-1, self.num_sample])
