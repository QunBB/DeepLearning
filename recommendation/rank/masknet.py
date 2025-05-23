"""
论文：MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

地址：https://arxiv.org/pdf/2102.0761
"""
import tensorflow as tf
from typing import List, Union, Optional
from functools import partial

from ..utils.core import dnn_layer


class MaskNet:
    def __init__(self,
                 agg_dim: int,
                 num_mask_block: int,
                 mask_block_ffn_size: Union[List[int], int],
                 masknet_type: str,
                 hidden_layer_size: Optional[List[int]] = None,
                 dropout: float = 0.,
                 l2_reg: float = 0.
                 ):
        """

        :param agg_dim: Instance-Guided Mask中Aggregation模块的输出维度
        :param num_mask_block: 串行结构中MaskBlock的层数 or 并行结构中MaskBlock的数量
        :param mask_block_ffn_size: 每一层MaskBlock的输出维度
        :param masknet_type: serial(串行)或parallel(并行)
        :param hidden_layer_size: 并行结构中每一层隐藏层的输出维度
        :param dropout:
        :param l2_reg:
        """
        self.l2_reg = l2_reg

        if masknet_type == 'serial':
            self.net_func = SerialMaskNet(agg_dim, num_mask_block, mask_block_ffn_size, hidden_layer_size, dropout, l2_reg)
            assert isinstance(mask_block_ffn_size, list) and len(mask_block_ffn_size) == num_mask_block
        elif masknet_type == 'parallel':
            self.net_func = ParallelMaskNet(agg_dim, num_mask_block, mask_block_ffn_size, hidden_layer_size, dropout, l2_reg)
            assert isinstance(mask_block_ffn_size, int) and isinstance(hidden_layer_size, list)
        else:
            raise TypeError('masknet_type only support "serial" or "parallel"')

    def __call__(self,
                 embeddings: Union[List[tf.Tensor], tf.Tensor],
                 is_training: bool = True):
        """

        :param embeddings: [bs, num_feature, dim] or list of [bs, dim]
        :param is_training:
        :return:
        """
        output = self.net_func(embeddings, is_training)

        output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer())
        return tf.reshape(output, [-1])


class MaskBlock:
    def __init__(self,
                 agg_dim: int,
                 num_mask_block: int,
                 mask_block_ffn_size: Union[List[int], int],
                 hidden_layer_size: Optional[List[int]] = None,
                 dropout: float = 0.,
                 l2_reg: float = 0.
                 ):
        self.agg_dim = agg_dim
        self.num_mask_block = num_mask_block
        self.mask_block_ffn_size = mask_block_ffn_size
        self.hidden_layer_size = hidden_layer_size
        self.l2_reg = l2_reg

        self.dnn_layer = partial(dnn_layer, dropout=dropout, use_bn=False, l2_reg=l2_reg)

    def embedding_norm(self, embeddings):
        if isinstance(embeddings, list):
            embeddings = tf.stack(embeddings, axis=1)

        assert len(embeddings.shape) == 3

        ln_embeddings = tf.contrib.layers.layer_norm(inputs=embeddings,
                                                     begin_norm_axis=-1,
                                                     begin_params_axis=-1)

        embeddings = tf.layers.flatten(embeddings)
        ln_embeddings = tf.layers.flatten(ln_embeddings)

        return embeddings, ln_embeddings

    def serial_model(self, embeddings, ln_embeddings, is_training):
        """串行MaskNet"""
        output = ln_embeddings
        for i in range(self.num_mask_block):
            mask = self.instance_guided_mask(embeddings, is_training, output_size=output.shape.as_list()[-1])
            output = self.mask_block(output, mask, self.mask_block_ffn_size[i], is_training)

        return output

    def parallel_model(self, embeddings, ln_embeddings, is_training):
        """并行MaskNet"""
        output_list = []
        for i in range(self.num_mask_block):
            mask = self.instance_guided_mask(embeddings, is_training)
            output = self.mask_block(ln_embeddings, mask, self.mask_block_ffn_size, is_training)
            output_list.append(output)

        final_output = self.dnn_layer(tf.concat(output_list, axis=-1), self.hidden_layer_size, activation=tf.nn.relu,
                                      is_training=is_training)
        return final_output

    def instance_guided_mask(self, embeddings, is_training, output_size=None):
        if output_size is None:
            output_size = embeddings.shape.as_list()[-1]
        agg = self.dnn_layer(embeddings, self.agg_dim, activation=tf.nn.relu, is_training=is_training)
        project = self.dnn_layer(agg, output_size, is_training=is_training)
        return project

    def mask_block(self, inputs, mask, output_size, is_training):
        masked = inputs * mask
        output = self.dnn_layer(masked, output_size, is_training=is_training)
        output = tf.contrib.layers.layer_norm(inputs=output,
                                              begin_norm_axis=-1,
                                              begin_params_axis=-1)
        return tf.nn.relu(output)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SerialMaskNet(MaskBlock):
    def __call__(self, embeddings, is_training=True):
        embeddings, ln_embeddings = self.embedding_norm(embeddings)

        output = self.serial_model(embeddings, ln_embeddings, is_training)

        return output


class ParallelMaskNet(MaskBlock):
    def __call__(self, embeddings, is_training=True):
        embeddings, ln_embeddings = self.embedding_norm(embeddings)

        output = self.parallel_model(embeddings, ln_embeddings, is_training)

        return output