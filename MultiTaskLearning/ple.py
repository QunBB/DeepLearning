import tensorflow as tf
from functools import reduce

import tensorflow.contrib.slim as slim


class PLE:
    def __init__(self, target_dict: dict,
                 num_experts: int,
                 num_levels: int,
                 experts_layer_size: list,
                 tower_layer_size: list,
                 l2_reg: float,
                 dropout: float):
        """

        :param target_dict: 多目标的分类标签数量，如 {"click": 2, "like": 2}
        :param num_experts: Experts的数量，这里所有task和共享的expert数量设置相同，可以根据实际情况进行设置不同的数量
        :param num_levels: extraction_network的层数
        :param experts_layer_size: 每一层extraction_network的expert维度, 如 [512]
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

    def __call__(self, inputs, is_training):
        # 多层的extraction_network
        ple_layer = {}
        with tf.variable_scope("PLE"):
            experts = self.extraction_network(inputs, is_training=is_training)

            assert len(experts) == len(self.target_dict)
            for name in self.target_dict:
                ple_layer[name] = experts[name]

        # tower层输出每个task的logits
        with tf.variable_scope("tower_layer"):
            tower_layer = {}
            for name in self.target_dict.keys():
                tower_layer[name] = self._mlp_layer(ple_layer[name], self.tower_layer_size,
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

    def extraction_network(self, inputs, is_training):
        """
        兼容单层和多层的PLE
        :param inputs: 原始的输入
        :param is_training:
        :return:
        """
        # 第一层的输入是模型的原始输入
        outputs = {name: inputs for name in list(self.target_dict.keys()) + ["shared"]}
        # 其他层的话，任务k的独享expert融合了上一层网络中任务k的独享expert和共享expert，而共享expert则融合了上一层所有的expert

        for level in range(self.num_levels):

            # 生成多个experts
            with tf.variable_scope("Mixture-of-Experts"):
                mixture_experts = {}
                for name in list(self.target_dict.keys()) + ["shared"]:
                    # 除了共享的expert，每个task拥有自己的expert
                    for i in range(self.num_experts):
                        # expert一般是一层全连接层
                        expert_layer = self._mlp_layer(outputs[name],
                                                       sizes=[self.experts_layer_size[level]],
                                                       is_training=is_training,
                                                       l2_reg=self.l2_reg,
                                                       dropout=self.dropout,
                                                       use_bn=True,
                                                       scope="{}_expert_{}_level_{}".format(name, i, level))
                        mixture_experts.setdefault(name, []).append(expert_layer)

            # 生成不同'输出expert'或task的gate
            with tf.variable_scope("Multi-gate"):
                multi_gate = {}
                for name in list(self.target_dict.keys()) + ["shared"]:
                    # 每个任务拥有独立一个gate
                    # 任务k的独享expert和共享expert融合作为下一层的输入，而共享expert则融合了所有的expert
                    # 因此，共享expert的gate的维度为 [batch_size, num_experts*(num_labels+1)]
                    # 而任务expert的gate的维度为 [batch_size, num_experts*2]
                    if name == 'shared':
                        gate_dim = self.num_experts*(len(self.target_dict)+1)
                    else:
                        gate_dim = self.num_experts * 2
                    gate = tf.layers.dense(inputs, units=gate_dim,
                                           kernel_initializer=slim.variance_scaling_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                           name="gate_{}_level_{}".format(name, level))
                    gate = tf.nn.softmax(gate)
                    multi_gate[name] = gate

            # 任务k的独享expert和共享expert融合作为下一层的输入，而共享expert则融合了所有的expert
            # 通过自己gate的权重分布进行expert融合
            with tf.variable_scope("combine_gate_expert"):
                ple_layer = {}
                for name in list(self.target_dict.keys()) + ["shared"]:
                    if name == 'shared':
                        # 最后一层是每个task的最终输出，无需合并共享expert
                        if level == self.num_levels - 1:
                            continue
                        merge_experts = []
                        for _name in list(self.target_dict.keys()) + ["shared"]:
                            merge_experts.extend(mixture_experts[_name])
                    else:
                        merge_experts = mixture_experts[name] + mixture_experts['shared']

                    ple_layer[name] = self._combine_expert_gate(merge_experts, multi_gate[name])

            outputs = ple_layer
        return outputs

    def _combine_expert_gate(self, mixture_experts, gate):
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
    import numpy as np
    model = PLE(target_dict={"click": 2, "like": 2},
                num_experts=5,
                num_levels=2,
                experts_layer_size=[1024, 512],
                tower_layer_size=[256, 128],
                l2_reg=0.00001,
                dropout=0.3)
    inputs = tf.placeholder(tf.float32, shape=[None, 2056], name='model_inputs')

    logits, pred = model(inputs, is_training=True)

    print(logits, "\n", pred)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([logits, pred], feed_dict={inputs: np.random.random([6, 2056])}))
