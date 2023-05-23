import tensorflow as tf
from typing import List, Optional, Callable
from typing import Dict as OrderedDictType
from functools import partial

from .fms import LinearTerms, FMType, Field, FMs
from ..Utils.core import dnn_layer


class DeepFM:
    def __init__(self,
                 fields_list: List[Field],
                 embedding_dim: int,
                 dnn_hidden_size: List[int],
                 dnn_activation: Optional[Callable] = None,
                 dnn_dropout: Optional[float] = 0.,
                 dnn_use_bn: Optional[bool] = True,
                 dnn_l2_reg: float = 0.,
                 linear_type: LinearTerms = LinearTerms.LW,
                 model_type: FMType = FMType.FM,
                 emb_l2_reg: float = 0.
                 ):
        self.fm = FMs(fields_list, embedding_dim, linear_type, model_type, emb_l2_reg)

        self.dnn_layer = partial(dnn_layer, hidden_size=dnn_hidden_size, activation=dnn_activation,
                                 dropout=dnn_dropout, use_bn=dnn_use_bn, l2_reg=dnn_l2_reg)
        self.dnn_l2_reg = dnn_l2_reg

    def __call__(self, sparse_inputs_dict: OrderedDictType[str, tf.Tensor],
                 dense_inputs_dict: OrderedDictType[str, tf.Tensor],
                 is_training: bool = True):
        fm_logit = self.fm(sparse_inputs_dict, dense_inputs_dict, add_sigmoid=False)

        embedding_output = self.fm.get_embedding_output()
        embedding_output = tf.concat(embedding_output, axis=1)

        dnn_output = self.dnn_layer(embedding_output, is_training=is_training)
        dnn_logit = tf.layers.dense(dnn_output, 1,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.dnn_l2_reg),
                                    kernel_initializer=tf.glorot_normal_initializer())

        output = tf.nn.sigmoid(fm_logit + tf.squeeze(dnn_logit, axis=1))

        return output
