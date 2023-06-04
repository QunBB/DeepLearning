import itertools
from collections import OrderedDict
from typing import List, Union
from typing import Dict as OrderedDictType

import tensorflow as tf

from .type_declaration import Field, LinearTerms


class LinearEmbedding:
    """
    embedding layer + linear
    """
    def __init__(self,
                 fields_list: List[Field],
                 linear_type: LinearTerms = LinearTerms.LW):
        self.embeddings_table = {}
        self.weights_table = {}

        for field in fields_list:
            # embeddings 隐向量
            self.embeddings_table[field.name] = tf.get_variable('emb_' + field.name,
                                                                shape=[field.vocabulary_size, field.dim],
                                                                regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))

            # 线性项权重
            if linear_type == LinearTerms.LW:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[field.vocabulary_size],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))
            elif linear_type == LinearTerms.FeLV:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name,
                                                                 shape=[field.vocabulary_size, field.dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))
            else:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[1, field.dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg))

    def _get_linear_logit(self,
                          w: tf.Tensor,
                          i: Union[int, tf.Tensor] = 0,
                          x: Union[int, tf.Tensor] = 1,
                          v: Union[int, tf.Tensor] = 1
                          ):
        """线性项计算"""
        shape = w.shape.as_list()

        if len(shape) == 1:  # LM
            return tf.gather(w, i) * x
        elif len(shape) == 2 and shape[0] == 1:  # FiLV
            return tf.reduce_sum(w * v, axis=1) * x
        elif len(shape) == 2 and shape[0] > 1:  # FeLV
            return tf.reduce_sum(tf.gather(w, i) * v, axis=1) * x
        else:
            raise ValueError

    def __call__(self,
                 sparse_id_inputs_dict: OrderedDictType[str, tf.Tensor] = {},
                 dense_value_inputs_dict: OrderedDictType[str, tf.Tensor] = {},
                 as_dict: bool = False):
        linear_logit = []
        embeddings = OrderedDict()

        for name, x in sparse_id_inputs_dict.items():
            v = tf.nn.embedding_lookup(self.embeddings_table[name], x)
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], i=x, v=v))
            embeddings[name] = v

        for name, x in dense_value_inputs_dict.items():
            v = tf.reshape(self.embeddings_table[name][0], [1, -1])
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], x=x, v=v))
            embeddings[name] = v * tf.reshape(x, [-1, 1])

        linear_logit = tf.add_n(linear_logit)

        if not as_dict:
            embeddings = list(embeddings.values())

        return embeddings, linear_logit


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


class SENet:
    """
    SENet+ Layer，支持不同field embeddings的size不等
    """

    def __init__(self,
                 reduction_ratio: int,
                 num_groups: int):
        self.reduction_ratio = reduction_ratio
        self.num_groups = num_groups

    def __call__(self, embeddings_list: List[tf.Tensor]):
        """

        :param embeddings_list: [embedding_1,...,embedding_i,...,embedding_f]，f为field的数目，embedding_i is [bs, dim]
        :return:
        """
        for emb in embeddings_list:
            assert len(emb.shape.as_list()) == 2, 'field embeddings must be rank 2 tensors'

        field_size = len(embeddings_list)
        feature_size_list = [emb.shape.as_list()[-1] for emb in embeddings_list]

        # Squeeze
        group_embeddings_list = [tf.reshape(emb, [-1, self.num_groups, tf.shape(emb)[-1] // self.num_groups]) for emb in
                                 embeddings_list]
        Z = [tf.reduce_mean(emb, axis=-1) for emb in group_embeddings_list] + [tf.reduce_max(emb, axis=-1) for emb in
                                                                               group_embeddings_list]
        Z = tf.concat(Z, axis=1)  # [bs, field_size * num_groups * 2]

        # Excitation
        reduction_size = max(1, field_size * self.num_groups * 2 // self.reduction_ratio)

        A_1 = tf.layers.dense(Z, reduction_size,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              activation=tf.nn.relu,
                              name='W_1')
        A_2 = tf.layers.dense(A_1, sum(feature_size_list),
                              kernel_initializer=tf.glorot_normal_initializer(),
                              activation=tf.nn.relu,
                              name='W_2')

        # Re-weight
        senet_like_embeddings = [emb * w for emb, w in zip(embeddings_list, tf.split(A_2, feature_size_list, axis=1))]

        # Fuse
        output = tf.concat(senet_like_embeddings, axis=-1) + tf.concat(embeddings_list, axis=-1)
        # Layer Normalization
        output = tf.contrib.layers.layer_norm(
            inputs=output, begin_norm_axis=-1, begin_params_axis=-1, scope='LN')

        return output
