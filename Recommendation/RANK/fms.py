"""
论文：Factorization Machines
地址：https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

论文：Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising
地址：https://arxiv.org/pdf/1806.03514.pdf

论文：Field-Embedded Factorization Machines for Click-through rate prediction
地址：https://arxiv.org/pdf/2009.09931.pdf
"""
import tensorflow as tf
from typing import List, Union
from typing import Dict as OrderedDictType
from dataclasses import dataclass
from enum import IntEnum
import itertools


class LinearTerms(IntEnum):
    """FwFMs中的线性项"""
    LW = 0
    FeLV = 1
    FiLV = 2


class FMType(IntEnum):
    """FMs选项"""
    FM = 1
    FwFM = 2
    FEFM = 3


@dataclass
class Field:
    name: str
    vocabulary_size: int = 1  # dense类型为1


class FMs:
    def __init__(self,
                 fields_list: List[Field],
                 embedding_dim: int,
                 linear_type: LinearTerms = LinearTerms.LW,
                 model_type: FMType = FMType.FM,
                 l2_reg: float = 0.):
        self.embeddings_table = {}
        self.weights_table = {}
        self.num_fields = len(fields_list)

        for field in fields_list:
            # embeddings 隐向量
            self.embeddings_table[field.name] = tf.get_variable('emb_' + field.name,
                                                                shape=[field.vocabulary_size, embedding_dim],
                                                                regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

            # 线性项权重
            if linear_type == LinearTerms.LW:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[field.vocabulary_size],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            elif linear_type == LinearTerms.FeLV:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name,
                                                                 shape=[field.vocabulary_size, embedding_dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            else:
                self.weights_table[field.name] = tf.get_variable('w_' + field.name, shape=[1, embedding_dim],
                                                                 regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

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

    def __call__(self,
                 sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 add_sigmoid: bool = True):
        """

        :param sparse_inputs_dict: 离散特征，经过LabelEncoder之后的输入
        :param dense_inputs_dict:
        :return:
        """
        linear_logit = []
        interactions = []

        for name, x in sparse_inputs_dict.items():
            v = tf.nn.embedding_lookup(self.embeddings_table[name], x)
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], i=x, v=v))
            interactions.append(v)

        for name, x in dense_inputs_dict.items():
            v = tf.reshape(self.embeddings_table[name][0], [1, -1])
            linear_logit.append(self._get_linear_logit(w=self.weights_table[name], x=x, v=v))
            interactions.append(v * tf.reshape(x, [-1, 1]))

        self._embedding_output = interactions

        fms_logit = self.interaction_func(interactions)

        final_logit = tf.add_n(linear_logit + [fms_logit])

        if add_sigmoid:
            final_logit = tf.nn.sigmoid(final_logit)

        return final_logit

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
