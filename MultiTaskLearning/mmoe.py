from typing import List, Dict
import tensorflow as tf
from functools import reduce

import tensorflow.contrib.slim as slim


class MMoE:
    def __init__(self, target_dict: Dict[str, int],
                 num_experts: int,
                 num_levels: int,
                 experts_layer_size: List[int],
                 tower_layer_size: List[int],
                 l2_reg: float,
                 dropout: float):
        """

        :param target_dict: 多目标的分类标签数量，如 {"click": 2, "like": 2}
        :param num_experts: Mixture of Experts, Experts的数量
        :param num_levels: MMoE的层数
        :param experts_layer_size: 每一层MMoE的expert维度, 如 [512]
        :param tower_layer_size: tower全连接层的维度, 如 [256, 128]
        :param l2_reg: 正则惩罚项
        :param dropout:
        """
        assert num_levels == len(experts_layer_size), "num_levels must be equal to the size of experts_layer_size"

        self.target_dict = target_dict
        self.num_experts = num_experts
        self.num_levels = num_levels
        self.experts_layer_size = experts_layer_size
        self.tower_layer_size = tower_layer_size
        self.l2_reg = l2_reg
        self.dropout = dropout

    def __call__(self,
                 inputs: tf.Tensor,
                 is_training: bool):
        # 多层的MMoE
        mmoe_layer = {}
        with tf.variable_scope("MMoE"):
            experts = self._moe_layer(inputs, is_training=is_training)

            assert len(experts) == len(self.target_dict)
            for name, one_expert in zip(self.target_dict.keys(), experts):
                mmoe_layer[name] = one_expert

        # tower层输出每个task的logits
        with tf.variable_scope("tower_layer"):
            tower_layer = {}
            for name in self.target_dict.keys():
                tower_layer[name] = self._mlp_layer(mmoe_layer[name], self.tower_layer_size,
                                                    is_training=is_training,
                                                    l2_reg=self.l2_reg,
                                                    dropout=self.dropout,
                                                    use_bn=True,
                                                    scope="tower_{}".format(name))
        # 计算每个task的预测
        with tf.variable_scope("prediction"):
            pred = {}
            logits = {}
            for name in self.target_dict.keys():
                output = tf.layers.dense(tower_layer[name], self.target_dict[name])
                logits[name] = tf.nn.softmax(output)

                pred[name] = tf.argmax(logits[name], axis=-1)

        return logits, pred

    def _moe_layer(self, inputs, is_training):
        """
        兼容单层和多层的MMoE
        :param inputs: 原始的输入
        :param is_training:
        :return:
        """
        # 第一层的输入是模型的原始输入
        outputs = inputs

        for level in range(self.num_levels):
            # 如果不是第一层，那么输入是多个上层的输出expert组成的列表
            # 此时，需要进行fusion：一般是拼接、相乘、相加几种融合方式
            # 这里使用相加拼接相乘
            if isinstance(outputs, list):
                outputs = tf.concat([reduce(lambda x, y: x + y, outputs),
                                    reduce(lambda x, y: x * y, outputs)],
                                    axis=-1)

            # 生成多个experts
            with tf.variable_scope("Mixture-of-Experts"):
                mixture_experts = []
                for i in range(self.num_experts):
                    # expert一般是一层全连接层
                    expert_layer = self._mlp_layer(outputs,
                                                   sizes=[self.experts_layer_size[level]],
                                                   is_training=is_training,
                                                   l2_reg=self.l2_reg,
                                                   dropout=self.dropout,
                                                   use_bn=True,
                                                   scope="expert_{}_level_{}".format(i, level))
                    mixture_experts.append(expert_layer)

            # 如果是最后一层，那么gate的数量应该是task的数量
            # 其他层的话，gate的数量一般等于experts的数量
            if level == self.num_levels - 1:
                num_gates = len(self.target_dict)
            else:
                num_gates = self.num_experts

            # 生成不同'输出expert'或task的gate
            with tf.variable_scope("Multi-gate"):
                multi_gate = []
                for i in range(num_gates):
                    # 每个task拥有独立一个gate
                    # 每个gate的维度为 [batch_size, num_experts]
                    gate = tf.layers.dense(inputs, units=self.num_experts,
                                           kernel_initializer=slim.variance_scaling_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                           name="gate_{}_level_{}".format(i, level))
                    gate = tf.nn.softmax(gate)
                    multi_gate.append(gate)

            # 每个'输出expert'或task，通过自己gate的权重分布，合并expert
            with tf.variable_scope("combine_gate_expert"):
                mmoe_layer = []
                for i in range(num_gates):
                    mmoe_layer.append(self._combine_expert_gate(mixture_experts, multi_gate[i]))

            outputs = mmoe_layer
        return outputs

    def _combine_expert_gate(self,
                             mixture_experts: List[tf.Tensor],
                             gate: tf.Tensor):
        """
        多个expert通过gate进行合并
        :param mixture_experts: 多个experts的list
        :param gate: 当前task的gate
        :return:
        """
        # [ [batch_size, dim], ....] -> [ [batch_size, 1, dim], ....] -> [batch_size, num, dim]
        mixture_experts = tf.concat([tf.expand_dims(dnn, axis=1) for dnn in mixture_experts], axis=1)
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
            with tf.variable_scope(scope+"_"+str(i)):
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
    model = MMoE(target_dict={"click": 2, "like": 2},
                 num_experts=5,
                 num_levels=2,
                 experts_layer_size=[1024, 512],
                 tower_layer_size=[256, 128],
                 l2_reg=0.00001,
                 dropout=0.3)
    inputs = tf.placeholder(tf.float32, shape=[None, 2056], name='model_inputs')

    logits, pred = model(inputs, is_training=True)

    print(logits, pred)
