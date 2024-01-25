"""
> 论文：xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
>
> 地址：https://arxiv.org/pdf/1803.05170.pdf
"""
import tensorflow as tf
from typing import Optional, Callable, List, Union
from typing import Dict as OrderedDictType
from functools import partial

from ..utils.core import dnn_layer
from ..utils.interaction import LinearEmbedding
from ..utils.type_declaration import Field, LinearTerms


class xDeepFM:
    def __init__(self,
                 fields_list: List[Field],
                 cross_layer_sizes: List[int],
                 dnn_hidden_units: List[int],
                 linear_type: LinearTerms = LinearTerms.LW,
                 low_rank_dim: Optional[int] = None,
                 split_connect: bool = False,
                 cross_activation: Optional[Callable] = None,
                 residual: bool = False,
                 dropout: float = 0.,
                 l2_reg: float = 0.,
                 dnn_activation: Optional[Callable] = None,
                 dnn_use_bn: bool = True,
                 ):
        """参数见CIN类"""
        self.field_embedding_nums = len(fields_list)

        self.linear = LinearEmbedding(fields_list, linear_type)

        self.cross_layer = CIN(self.field_embedding_nums, cross_layer_sizes, low_rank_dim, split_connect, cross_activation,
                               residual, dnn_activation, dropout, l2_reg)

        self.dnn_layer = partial(dnn_layer, hidden_units=dnn_hidden_units, activation=dnn_activation,
                                 dropout=dropout, use_bn=dnn_use_bn, l2_reg=l2_reg)
        self.l2_reg = l2_reg

        self.linear_w = tf.get_variable('linear_w', shape=[1], initializer=tf.ones_initializer())
        self.cin_w = tf.get_variable('cin_w', shape=[1], initializer=tf.ones_initializer())
        self.dnn_w = tf.get_variable('dnn_w', shape=[1], initializer=tf.ones_initializer())
        self.bias = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 is_training: bool = True):
        """
        未经过embedding layer的输入
        :param sparse_inputs_dict: 离散特征，经过LabelEncoder之后的输入
        :param dense_inputs_dict: 连续值特征
        :return:
        """
        assert len(sparse_inputs_dict) + len(dense_inputs_dict) == self.field_embedding_nums

        embeddings, linear_logit = self.linear(sparse_inputs_dict, dense_inputs_dict)

        cin_logit = self.cross_layer(embeddings, is_training)
        cin_logit = tf.reshape(cin_logit, [-1])

        dnn_output = self.dnn_layer(tf.concat(embeddings, axis=-1), is_training=is_training)
        dnn_logit = tf.layers.dense(dnn_output, 1,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                    kernel_initializer=tf.glorot_normal_initializer())
        dnn_logit = tf.reshape(dnn_logit, [-1])

        logit = linear_logit * self.linear_w + cin_logit * self.cin_w + dnn_logit * self.dnn_w + self.bias

        return tf.nn.sigmoid(logit)


class CIN:
    """define Factorization-Machine based Neural Network Model"""

    def __init__(self,
                 field_embedding_nums: int,
                 cross_layer_sizes: List[int],
                 low_rank_dim: Optional[int] = None,
                 split_connect: bool = False,
                 cross_activation: Optional[Callable] = None,
                 residual: bool = False,
                 residual_activation: Optional[Callable] = None,
                 dropout: float = 0.,
                 l2_reg: float = 0.
                 ):
        """

        :param field_embedding_nums: field的数目
        :param cross_layer_sizes: CIN的层数
        :param low_rank_dim: 参数矩阵应用low-rank的维度，None代表不使用
        :param split_connect: split=True则代表feature maps会平均拆成两份，一半connect到output unit，一半与next layer交叉。
        :param cross_activation: 交叉项的激活函数
        :param residual: 是否增加残差网络
        :param residual_activation: 残差网络的激活函数
        :param dropout:
        :param l2_reg:
        """
        self.field_embedding_nums = field_embedding_nums
        self.cross_layer_sizes = cross_layer_sizes
        self.split_connect = split_connect
        self.residual = residual
        self.cross_activation = cross_activation
        self.residual_activation = residual_activation
        self.dropout = dropout
        self.l2_reg = l2_reg

        field_nums = [field_embedding_nums]

        self.filters = []
        self.bias = []
        for idx, layer_size in enumerate(cross_layer_sizes):
            if low_rank_dim:
                filters0 = tf.get_variable("f0_" + str(idx),
                                           shape=[1, layer_size, field_nums[0], low_rank_dim],
                                           dtype=tf.float32,
                                           initializer=tf.glorot_normal_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                filters_ = tf.get_variable("f__" + str(idx),
                                           shape=[1, layer_size, low_rank_dim, field_nums[-1]],
                                           dtype=tf.float32,
                                           initializer=tf.glorot_normal_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                filters_m = tf.matmul(filters0, filters_)
                filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                filters = tf.transpose(filters_o, perm=[0, 2, 1])
            else:
                filters = tf.get_variable(name="f_" + str(idx),
                                          shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                          dtype=tf.float32,
                                          initializer=tf.glorot_normal_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.filters.append(filters)

            if split_connect:
                field_nums.append(int(layer_size / 2))
            else:
                field_nums.append(layer_size)

            b = tf.get_variable(name="f_b" + str(idx),
                                shape=[layer_size],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            self.bias.append(b)

    def __call__(self, nn_input: Union[tf.Tensor, List[tf.Tensor]],
                 is_training: bool = True):
        if isinstance(nn_input, list):
            nn_input = tf.stack(nn_input, axis=1)

        shape = nn_input.shape.as_list()
        assert len(shape) == 3 and shape[1] == self.field_embedding_nums
        dim = shape[2]

        hidden_nn_layers = []
        final_len = 0
        field_nums = [self.field_embedding_nums]

        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        with tf.variable_scope("exfm_part"):
            for idx, layer_size in enumerate(self.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                curr_out = tf.nn.conv1d(dot_result, filters=self.filters[idx], stride=1, padding='VALID')

                # BIAS ADD
                curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

                curr_out = self._activate_layer(curr_out, self.cross_activation)

                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                if not self.split_connect:
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(layer_size)

                else:
                    if idx != len(self.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size

                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)
            if self.residual:
                exFM_out0 = tf.layers.dense(result, 128, bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                            kernel_initializer=tf.glorot_normal_initializer()
                                            )
                exFM_out1 = self._activate_layer(exFM_out0, self.residual_activation,
                                                 dropout=self.dropout if is_training else None)
                exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
                exFM_out = tf.layers.dense(exFM_in, 1, bias_initializer=tf.zeros_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                           kernel_initializer=tf.glorot_normal_initializer())

            else:
                exFM_out = tf.layers.dense(result, 1, bias_initializer=tf.zeros_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                           kernel_initializer=tf.glorot_normal_initializer())

            return exFM_out

    def _activate_layer(self, logit, activation, dropout=None):
        if dropout:
            logit = tf.nn.dropout(logit, dropout)

        if activation:
            logit = activation(logit)

        return logit
