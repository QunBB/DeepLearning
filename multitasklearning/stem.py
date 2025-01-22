"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
from typing import List, Dict, Optional, Callable
from functools import partial
from collections import defaultdict, ChainMap

import tensorflow as tf
import tensorflow.contrib.slim as slim


class STEM:
    def __init__(self,
                 target_dict: Dict[str, Dict[str, int]],
                 auxiliary_target_dict: Optional[Dict[str, Dict[str, int]]] = None,
                 stop_gradients: bool = False,
                 experts_layer_size: List[int] = [128, 64],
                 tower_layer_size: List[int] = [128, 64],
                 activation: Callable = tf.nn.relu,
                 dropout: float = 0.,
                 l2_reg: float = 0.,
                 use_bn: bool = False):
        """Shared and Task-specific Embedding (STEM) and Asymmetric Multi-Embedding (AME)

        :param target_dict: 每一组任务的分类标签数量，如:
                        {
                        'group_1': {'click': 2, 'like': 2},
                        'group_2': {'fav': 2}
                        }
        :param auxiliary_target_dict: 同上，针对辅助任务
        :param stop_gradients: 是否对其他任务的experts取消反向传播
        :param experts_layer_size: expert DNN的维度, 如 [512]
        :param tower_layer_size: tower DNN的维度, 如 [512]
        :param activation: 激活函数
        :param dropout:
        :param l2_reg: 正则惩罚项
        :param use_bn: 是否使用batch_normalization
        """
        self.target_dict = target_dict
        self.auxiliary_target_dict = auxiliary_target_dict or {}
        self.stop_gradients = stop_gradients

        self.main_task = dict(ChainMap({}, *list((target_dict.values()))))
        self.auxiliary_task = dict(ChainMap({}, *list((auxiliary_target_dict.values()))))

        self.expert_mlp = partial(self._mlp_layer, sizes=experts_layer_size, l2_reg=l2_reg, dropout=dropout, activation=activation, use_bn=use_bn)
        self.tower_mlp = partial(self._mlp_layer, sizes=tower_layer_size, l2_reg=l2_reg, dropout=dropout, activation=activation, use_bn=use_bn)

    def __call__(self,
                 group_embeddings: Dict[str, tf.Tensor],
                 shared_embeddings: Optional[tf.Tensor] = None,
                 is_training: bool = True):
        """

        :param group_embeddings: 每一组经过embedding layer的输入，分组要与`target_dict`对应，同一个分组是共享同一个embedding table
        :param shared_embeddings: 共享expert的embedding输入，共享expert一般是独享一个embedding table
        :param is_training:
        :return:
        """
        for group in group_embeddings:
            assert group in self.target_dict or group in self.auxiliary_target_dict, f"当前分组 \"{group}\" 不存在"

        # 计算每个任务的experts
        task_experts = {}
        task_embeddings = {}
        for group, embeddings in group_embeddings.items():
            for task in self.target_dict.get(group) or self.auxiliary_target_dict.get(group):
                task_experts[task] = self.expert_mlp(inputs=embeddings, is_training=is_training, scope=f"expert_{task}")
                task_embeddings[task] = embeddings

        # 每一个任务分配experts
        assign_experts = defaultdict(list)
        for task_i in self.main_task:
            for task_j in task_experts:
                if self.stop_gradients and task_i != task_j:
                    assign_experts[task_i].append(tf.stop_gradient(task_experts[task_j]))
                else:
                    assign_experts[task_i].append(task_experts[task_j])
        for task in self.auxiliary_task:
            assign_experts[task] = [task_experts[task]]
        if shared_embeddings is not None:
            for task in assign_experts:
                assign_experts[task].append(shared_embeddings)

        # 计算每一个task的tower层和输出
        pred = {}
        logit = {}
        for task in assign_experts:
            tower_inputs = self._combine_expert_gate(assign_experts[task], task_embeddings[task])
            tower_output = self.tower_mlp(inputs=tower_inputs, is_training=is_training, scope=f"tower_{task}")

            logit[task] = tf.layers.dense(tower_output, self.main_task.get(task) or self.auxiliary_task[task],
                                          activation=tf.nn.softmax,
                                          kernel_initializer=slim.variance_scaling_initializer())
            pred[task] = tf.argmax(logit[task], axis=-1)

        return pred, logit

    def _combine_expert_gate(self,
                             mixture_experts: List[tf.Tensor],
                             embeddings: tf.Tensor):
        """
        多个expert通过gate进行合并
        :param mixture_experts: 多个experts的list
        :param gate: 当前task的gate
        :return:
        """
        # [ [batch_size, dim], ....] -> [batch_size, n, dim]
        mixture_experts = tf.stack(mixture_experts, axis=1)
        gate = tf.layers.dense(embeddings, units=mixture_experts.shape[1],
                               kernel_initializer=slim.variance_scaling_initializer())
        # [batch_size, num, 1]
        gate = tf.expand_dims(gate, axis=-1)
        # [batch_size, dim]
        return tf.reduce_sum(mixture_experts * gate, axis=1)

    def _mlp_layer(self, inputs, sizes, is_training,
                   l2_reg=0., dropout=0., use_bn=False, activation=tf.nn.relu, scope=None):
        """
        标准的MLP网络层
        :param inputs:
        :param sizes: 全连接的维度，如 [256, 128]
        :param is_training: 当前是否为训练阶段
        :param l2_reg: 正则惩罚项
        :param dropout:
        :param use_bn: 是否使用batch_normalization
        :param activation: 激活函数
        :return:
        """
        output = None

        for i, units in enumerate(sizes):
            with tf.variable_scope(scope + "_" + str(i)):
                output = tf.layers.dense(inputs, units=units,
                                         kernel_initializer=slim.variance_scaling_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

                if use_bn:
                    output = tf.layers.batch_normalization(output, training=is_training)

                if activation is not None:
                    output = activation(output)

                if is_training:
                    output = tf.nn.dropout(output, 1 - dropout)

        return output


if __name__ == '__main__':
    # 测试代码
    import numpy as np

    batch_size = 32
    dim = 128

    main_task_group = {
        'group_1': {'click': 2, 'like': 2},
        'group_2': {'fav': 2}
    }
    auxiliary_task_group = {
        'group_3': {'comment': 2}
    }

    model = STEM(main_task_group, auxiliary_task_group)

    pred, logit = model(
        group_embeddings={'group_1': np.random.random([batch_size, dim]),
                          'group_2': np.random.random([batch_size, dim]),
                          'group_3': np.random.random([batch_size, dim]), }
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(pred))
    print(sess.run(logit))