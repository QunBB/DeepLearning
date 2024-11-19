import tensorflow as tf
from typing import List, Callable, Optional, Union

from tensorflow.python.keras import activations


def dnn_layer(inputs: tf.Tensor,
              hidden_units: Union[List[int], int],
              activation: Optional[Union[Callable, str]] = None,
              dropout: Optional[float] = 0.,
              is_training: Optional[bool] = True,
              use_bn: Optional[bool] = True,
              l2_reg: float = 0.,
              use_bias: bool = True,
              scope=None):
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    output = inputs
    for idx, size in enumerate(hidden_units):
        output = tf.layers.dense(output, size,
                                 use_bias=use_bias,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 name=scope+f'_{idx}' if scope else None)
        if use_bn:
            output = tf.layers.batch_normalization(output, training=is_training, name=scope+f'_bn_{idx}' if scope else None)

        if activation is not None:
            output = activation_layer(activation, is_training=is_training, scope=f'activation_layer_{idx}')(output)

        if is_training:
            output = tf.nn.dropout(output, 1 - dropout)

    return output


def activation_layer(activation: Union[Callable, str],
                     scope: Optional[str] = None,
                     is_training: bool = True):
    if isinstance(activation, str):
        if activation.lower() == 'dice':
            return lambda x: dice(x, is_training, scope if scope else '')
        elif activation.lower() == 'prelu':
            return lambda x: prelu(x, scope if scope else '')
        else:
            return activations.get(activation)
    else:
        if activation is dice:
            return lambda x: dice(x, is_training, scope if scope else '')
        elif activation is prelu:
            return lambda x: prelu(x, scope if scope else '')
        else:
            return activation


def dice(_x, is_training, name=''):
    with tf.variable_scope(name_or_scope=name):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta', _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)

    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, training=is_training)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def prelu(_x, scope=''):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
