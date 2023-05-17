import tensorflow as tf
from typing import List, Callable, Optional


def dnn_layer(inputs: tf.Tensor,
              hidden_size: List[int],
              activation: Optional[Callable] = None,
              dropout: Optional[float] = 0.,
              is_training: Optional[bool] = True,
              use_bn: Optional[bool] = True,
              l2_reg: float = 0.):
    output = inputs
    for size in hidden_size:
        output = tf.layers.dense(output, size,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())
        if use_bn:
            output = tf.layers.batch_normalization(output, training=is_training)

        if activation is not None:
            output = activation(output)

        if is_training:
            output = tf.nn.dropout(output, 1 - dropout)

    return output
