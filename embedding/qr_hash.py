from typing import Optional, Union
import tensorflow as tf


class QRHashEmbedding:

    def __init__(self,
                 dim: int,
                 origin_num: int,
                 remainder_num: int,
                 compositional_type: str = 'multiply',
                 hashing: bool = True,
                 l2_reg: Optional[float] = None):
        """

        :param dim: 单个hash embedding维度
        :param origin_num: QR之前的ID数量
        :param remainder_num: 取模操作的模数
        :param compositional_type: hash embeddings的组合方式
        :param hashing: 是否需要对输入进行hash id映射
        :param l2_reg: embedding的正则惩罚
        """
        self.origin_num = origin_num
        self.remainder_num = remainder_num

        quotient_num = origin_num // remainder_num
        self.quotient_embedding = tf.get_variable('quotient_embedding',
                                                  shape=[quotient_num, dim],
                                                  initializer=tf.random_uniform_initializer,
                                                  regularizer=tf.contrib.layers.l2_regularizer(l2_reg) if l2_reg is not None else None)
        self.remainder_embedding = tf.get_variable('remainder_embedding',
                                                   shape=[self.remainder_num, dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer,
                                                   regularizer=tf.contrib.layers.l2_regularizer(l2_reg) if l2_reg is not None else None)

        self.compositional_type = compositional_type
        if compositional_type == 'multiply':
            self.compositional_func = tf.multiply
        elif compositional_type == 'concat':
            self.compositional_func = lambda *args: tf.concat(args, axis=-1)
        elif compositional_type == 'add':
            self.compositional_func = tf.add
        else:
            raise TypeError(f'Only support `compositional_type`: "multiply", "concat", "add"')

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
                # 重新设为0，是为了兼容embedding lookup
                inputs = tf.maximum(inputs, 0)

        # QR Trick
        q_index = inputs // self.remainder_num
        r_index = inputs % self.remainder_num
        q_embs = tf.nn.embedding_lookup(self.quotient_embedding, q_index)
        r_embs = tf.nn.embedding_lookup(self.remainder_embedding, r_index)

        # compositional embeddings
        compositional_embs = self.compositional_func(q_embs, r_embs)
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

        hash_index = tf.strings.to_hash_bucket_fast(inputs, self.origin_num)

        return hash_index, mask
