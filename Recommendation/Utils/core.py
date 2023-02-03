import tensorflow as tf
from typing import List, Callable


def dnn_layer(inputs,
              hidden_size: List[int],
              activation: Callable,
              dropout: float,
              is_training: bool,
              use_bn: bool,
              l2_reg: float = 0.):
    output = inputs
    for size in hidden_size:
        output = tf.layers.dense(output, size,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())
        if use_bn:
            output = tf.layers.batch_normalization(output)

        output = activation(output)

        if is_training:
            output = tf.nn.dropout(output, 1 - dropout)

    return output
