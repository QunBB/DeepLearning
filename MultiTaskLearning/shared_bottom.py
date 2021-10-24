import tensorflow as tf

import tensorflow.contrib.slim as slim


class BaseMTL:

    def __init__(self, target_dict: dict,
                 sharing_layer_size: list,
                 expert_layer_size: list,
                 l2_reg: float,
                 dropout: float):
        """

        :param target_dict: 多目标的分类标签数量，如 {"click": 2, "like": 2}
        :param sharing_layer_size: 共享层的维度, 如 [512]
        :param expert_layer_size: 专家层的维度, 如 [256, 128]
        :param l2_reg: 正则惩罚项
        :param dropout:
        """
        self.target_dict = target_dict
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.sharing_layer_size = sharing_layer_size
        self.expert_layer_size = expert_layer_size

    def __call__(self, inputs, is_training):
        """
        MTL网络层
        :param inputs: 输入为 经过embedding层之后的特征(拼接dense特征)
        :param is_training: 当前是否为训练阶段
        :return:
        """
        with tf.variable_scope("share-bottom"):
            sharing_layer = self._mlp_layer(inputs, self.sharing_layer_size, is_training=is_training,
                                            l2_reg=self.l2_reg, dropout=self.dropout, use_bn=True)

        with tf.variable_scope("expert_layer"):
            expert_layer = {}
            for name in self.target_dict.keys():
                expert_layer[name] = self._mlp_layer(sharing_layer, self.expert_layer_size, is_training=is_training,
                                                     l2_reg=self.l2_reg, dropout=self.dropout, use_bn=True)

        with tf.variable_scope("prediction"):
            pred = {}
            logits = {}
            for name in self.target_dict.keys():
                output = tf.layers.dense(expert_layer[name], self.target_dict[name])
                logits[name] = tf.nn.softmax(output)

                pred[name] = tf.argmax(logits[name])

        return logits, pred

    def _mlp_layer(self, inputs, sizes, is_training,
                   l2_reg=0., dropout=0., use_bn=False, activation=tf.nn.relu):
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

        for units in sizes:
            output = tf.layers.dense(inputs, units=units,
                                     kernel_initializer=slim.variance_scaling_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
                                     )

            if use_bn:
                output = tf.layers.batch_normalization(output, training=is_training)

            if activation is not None:
                output = activation(output)

            if is_training:
                output = tf.nn.dropout(output, 1 - dropout)

        return output
