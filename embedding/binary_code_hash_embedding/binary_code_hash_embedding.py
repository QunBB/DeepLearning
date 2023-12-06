"""
论文：Binary Code based Hash Embedding forWeb-scale Applications

地址：https://arxiv.org/pdf/2109.02471.pdf
"""
from typing import Optional, Union
import math
import tensorflow as tf


class BinaryCodeHashEmbedding:
    def __init__(self,
                 dim: int,
                 length: int,
                 t: int,
                 strategy: str,
                 compositional_type: str = 'pooling',
                 hashing: bool = True,
                 l2_reg: Optional[float] = None,
                 op_library_path: str = './tensorflow_binary_code_hash/python/ops/_binary_code_hash_ops.so'):
        """

        :param dim: 单个block embedding维度
        :param length: 二进制码的长度
        :param t: strategy="succession"时表示每t个0-1值分到同一block。strategy="skip"时表示间隔为t的0-1值放入同一个block中。详见论文
        :param strategy: "succession" or "skip"
        :param compositional_type: block embeddings的组合方式
        :param hashing: 是否需要对输入进行hash id映射
        :param l2_reg: embedding的正则惩罚
        :param op_library_path: 自定义算子编译的so文件位置
        """
        assert length <= 64, f"Need length <= 64, got length: {length}"
        self.hash_length = 2 ** length

        # block_length表示每个block中0-1的个数
        if strategy == 'succession':
            block_num = math.ceil(length / t)
            block_length = t
        elif strategy == 'skip':
            block_num = t + 1
            block_length = math.ceil(length / block_num)
        else:
            raise Exception(f'Strategy must be "succession" or "skip", but got strategy: {strategy}')

        # 引入自定义算子
        binary_code_hash_ops = tf.load_op_library(op_library_path)
        binary_code_hash = binary_code_hash_ops.binary_code_hash
        self.binary_code_hash_op = lambda x: binary_code_hash(x, length=length, t=t, strategy=strategy)
        # 所有block的embedding table进行合并
        # 因为自定义算子实现了第n个block的输出索引会加上(n-1)*2^k, k为每个block中0-1的个数
        self.block_embeddings = tf.get_variable('block_embeddings',
                                                shape=[block_num * (2 ** block_length), dim],
                                                initializer=tf.random_uniform_initializer,
                                                regularizer=tf.contrib.layers.l2_regularizer(l2_reg) if l2_reg is not None else None)

        self.compositional_type = compositional_type
        if compositional_type == 'pooling':
            self.compositional_func = lambda x: tf.reduce_sum(x, axis=-2)
        elif compositional_type == 'concat':
            def concat_func(x):
                shape = x.shape.as_list()
                return tf.reshape(x, shape[:-2] + [shape[-2] * shape[-1]])
            self.compositional_func = concat_func
        else:
            raise Exception(f'compositional_type must be "pooling" or "concat", but got compositional_type: {compositional_type}')

        self.hashing = hashing

    def __call__(self,
                 inputs: Union[tf.Tensor, tf.sparse.SparseTensor],
                 return_mask: bool):
        """

        :param inputs:
        :param return_mask: 是否需要返回SparseTensor的mask
        :return:
        """
        mask = None

        if not self.hashing and inputs.dtype not in (tf.int32, tf.int64):
            raise Exception('Inputs must be unique id with type `int32` or `int64` when not hashing')

        if self.hashing:
            inputs, mask = self.hash_function(inputs)
        else:
            if isinstance(inputs, tf.sparse.SparseTensor):
                # 默认值设为-1是为了能够区分缺失位置
                inputs = tf.sparse.to_dense(inputs, default_value=-1, validate_indices=True)
                mask = tf.cast(tf.math.not_equal(inputs, -1), tf.float32)
                # 重新设为0，是为了兼容 ID >= 0
                inputs = tf.maximum(inputs, 0)

        # Binary Code Hash ID
        binary_code_hash_id = self.binary_code_hash_op(inputs)
        # Binary Code Hash Embedding
        binary_code_hash_embedding = tf.nn.embedding_lookup(self.block_embeddings, binary_code_hash_id)

        # Compositional embeddings
        compositional_embs = self.compositional_func(binary_code_hash_embedding)
        if mask is not None:
            compositional_embs = tf.multiply(compositional_embs, tf.expand_dims(mask, axis=-1))

        if return_mask:
            return compositional_embs, mask
        else:
            return compositional_embs

    def hash_function(self, inputs):
        mask = None
        if isinstance(inputs, tf.sparse.SparseTensor):
            if inputs.dtype != tf.string:
                inputs = tf.sparse.to_dense(inputs, default_value=-1, validate_indices=True, name=None)
                inputs = tf.strings.as_string(inputs)
            else:
                inputs = tf.sparse.to_dense(inputs, default_value='-1', validate_indices=True, name=None)
            mask = tf.cast(tf.math.not_equal(inputs, '-1'), tf.float32)
        else:
            if inputs.dtype != tf.string:
                inputs = tf.strings.as_string(inputs)

        hash_index = tf.strings.to_hash_bucket_fast(inputs, self.hash_length)

        return hash_index, mask
