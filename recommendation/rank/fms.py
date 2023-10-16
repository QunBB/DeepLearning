"""
论文：Factorization Machines
地址：https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

论文：Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising
地址：https://arxiv.org/pdf/1806.03514.pdf

论文：Field-Embedded Factorization Machines for Click-through rate prediction
地址：https://arxiv.org/pdf/2009.09931.pdf
"""
import itertools
from typing import Dict as OrderedDictType
from typing import List

import tensorflow as tf

from ..utils.interaction import LinearEmbedding
from ..utils.type_declaration import LinearTerms, FMType, Field


class FMs:
    def __init__(self,
                 fields_list: List[Field],
                 linear_type: LinearTerms = LinearTerms.LW,
                 model_type: FMType = FMType.FM,
                 l2_reg: float = 0.):
        self.num_fields = len(fields_list)

        embedding_dim = fields_list[0].dim  # 所有field embeddings的维度应该相同

        if model_type == FMType.FwFM:
            self.interaction_strengths = tf.get_variable('interaction_strengths',
                                                         shape=[self.num_fields, self.num_fields],
                                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            self.interaction_func = self._fwfm_interaction
        elif model_type == FMType.FEFM:
            self.interaction_strengths = tf.get_variable('interaction_strengths',
                                                         shape=[self.num_fields, self.num_fields, embedding_dim, embedding_dim],
                                                         regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            self.interaction_func = self._fefm_interaction
        else:
            self.interaction_func = self._fm_interaction

        # For DeepFM
        self._embedding_output = []

        self.linear = LinearEmbedding(fields_list, linear_type)

        self.global_w = tf.get_variable('global_w', shape=[1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 add_sigmoid: bool = True):
        """
        未经过embedding layer的输入
        :param sparse_inputs_dict: 离散特征，经过LabelEncoder之后的输入
        :param dense_inputs_dict:
        :return:
        """
        assert self.num_fields == len(sparse_inputs_dict) + len(dense_inputs_dict)

        embeddings, linear_logit = self.linear(sparse_inputs_dict, dense_inputs_dict)

        self._embedding_output = embeddings

        fms_logit = self.interaction_func(embeddings)

        final_logit = linear_logit + fms_logit + self.global_w

        if add_sigmoid:
            final_logit = tf.nn.sigmoid(final_logit)

        return final_logit

    def _fm_interaction(self, interactions):
        interactions = tf.stack(interactions, axis=1)
        square_of_sum = tf.square(tf.reduce_sum(
            interactions, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            interactions * interactions, axis=1, keep_dims=True)
        fm_logit = square_of_sum - sum_of_square
        fm_logit = 0.5 * tf.reduce_sum(fm_logit, axis=2, keep_dims=False)

        return tf.reshape(fm_logit, [-1])

    def _fwfm_interaction(self, interactions):
        logits = []
        for i, j in itertools.combinations(range(self.num_fields), 2):
            r_ij = self.interaction_strengths[i, j]
            vx_i = interactions[i]
            vx_j = interactions[j]
            logits.append(tf.reduce_sum(r_ij * vx_i * vx_j, axis=1))

        return tf.add_n(logits)

    def _fefm_interaction(self, interactions):
        logits = []
        for i, j in itertools.combinations(range(self.num_fields), 2):
            w_ij = self.interaction_strengths[i, j]
            vx_i = interactions[i]
            vx_j = interactions[j]

            _logit = tf.matmul(tf.matmul(tf.expand_dims(vx_i, axis=1), w_ij),
                               tf.expand_dims(vx_j, axis=2))
            logits.append(tf.reshape(_logit, [-1]))

        return tf.add_n(logits)

    def get_embedding_output(self):
        return self._embedding_output
