import tensorflow as tf
from typing import List, Callable, Optional, Union


def dnn_layer(inputs: tf.Tensor,
              hidden_size: Union[List[int], int],
              activation: Optional[Callable] = None,
              dropout: Optional[float] = 0.,
              is_training: Optional[bool] = True,
              use_bn: Optional[bool] = True,
              l2_reg: float = 0.,
              use_bias: bool = True,
              scope=None):
    if isinstance(hidden_size, int):
        hidden_size = [hidden_size]

    output = inputs
    for size in hidden_size:
        output = tf.layers.dense(output, size,
                                 use_bias=use_bias,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 name=scope)
        if use_bn:
            output = tf.layers.batch_normalization(output, training=is_training, name=scope)

        if activation is not None:
            output = activation(output)

        if is_training:
            output = tf.nn.dropout(output, 1 - dropout)

    return output
