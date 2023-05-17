import itertools
from typing import List

import tensorflow as tf


class BiLinear:
    def __init__(self,
                 output_size: int,
                 bilinear_type: str,
                 equal_dim: bool = True,
                 bilinear_plus: bool = False,
                 ):
        """
        双线性特征交互层，支持不同field embeddings的size不等
        :param output_size: 输出的size
        :param bilinear_type: ['all', 'each', 'interaction']，支持其中一种
        :param equal_dim: 所有field embeddings的size是否相同
        :param bilinear_plus: 是否使用bi-linear+
        """
        self.bilinear_type = bilinear_type
        self.output_size = output_size

        if bilinear_type not in ['all', 'each', 'interaction']:
            raise NotImplementedError("bilinear_type only support: ['all', 'each', 'interaction']")

        # 当所有field embeddings的size不等时，bilinear_type只能为'interaction'
        if not equal_dim:
            self.bilinear_type = 'interaction'

        if bilinear_plus:
            self.func = self._full_interaction
        else:
            self.func = tf.multiply

    def __call__(self, embeddings_inputs: List[tf.Tensor]):
        field_size = len(embeddings_inputs)

        # field embeddings的size
        _dim = embeddings_inputs[0].shape.as_list()[-1]

        # bi-linear+: p的维度为[bs, f*(f-1)/2]
        # bi-linear:
        # 当equal_dim=True时，p的维度为[bs, f*(f-1)/2*k]，k为embeddings的size
        # 当equal_dim=False时，p的维度为[bs, (k_2+k_3+...+k_f)+...+(k_i+k_{i+1}+...+k_f)+...+k_f]，k_i为第i个field的embedding的size
        if self.bilinear_type == 'all':
            v_dot = [tf.layers.dense(v_i, _dim,
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     name='bilinear', reuse=tf.AUTO_REUSE)
                     for v_i in embeddings_inputs[:-1]]
            p = [self.func(v_dot[i], embeddings_inputs[j]) for i, j in itertools.combinations(range(field_size), 2)]
        elif self.bilinear_type == 'each':
            v_dot = [tf.layers.dense(v_i, _dim,
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     name=f'bilinear_{i}', reuse=tf.AUTO_REUSE)
                     for i, v_i in enumerate(embeddings_inputs[:-1])]
            p = [self.func(v_dot[i], embeddings_inputs[j])
                 for i, j in itertools.combinations(range(field_size), 2)]
        else:  # interaction
            p = [self.func(tf.layers.dense(embeddings_inputs[i], embeddings_inputs[j].shape.as_list()[-1],
                                           kernel_initializer=tf.glorot_normal_initializer(),
                                           name=f'bilinear_{i}_{j}', reuse=tf.AUTO_REUSE), embeddings_inputs[j])
                 for i, j in itertools.combinations(range(field_size), 2)]

        output = tf.layers.dense(tf.concat(p, axis=-1), self.output_size,
                                 kernel_initializer=tf.glorot_normal_initializer())
        return output

    def _full_interaction(self, v_i, v_j):
        # [bs, 1, dim] x [bs, dim, 1] = [bs, 1]
        interaction = tf.matmul(tf.expand_dims(v_i, axis=1), tf.expand_dims(v_j, axis=-1))
        return tf.reshape(interaction, [-1, 1])


if __name__ == '__main__':
    # test
    bileanr = BiLinear(64, 'each', equal_dim=True, bilinear_plus=False)
    output = bileanr([tf.placeholder(tf.float32, [None, 64]) for _ in range(10)]
                     # + [tf.placeholder(tf.float32, [None, 128]) for _ in range(20)]
                     )
    print(output)
