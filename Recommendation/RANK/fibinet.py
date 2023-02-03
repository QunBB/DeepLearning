import tensorflow as tf
from typing import List, Callable

from Recommendation.Utils.core import dnn_layer
from Recommendation.Utils.senet import SENet
from Recommendation.Utils.bilinear import BiLinear


class FiBiNet:
    def __init__(self,
                 dnn_units: List[int],
                 dropout: float,
                 reduction_ratio: int,
                 num_groups: int,
                 bilinear_output_size: int,
                 bilinear_type: str,
                 dnn_activation: Callable = tf.nn.relu,
                 dnn_use_bn: bool = True,
                 dnn_l2_reg: float = 0.,
                 bilinear_plus: bool = True,
                 equal_dim: bool = True):
        """
        FiBiNet和FiBiNet++模型，支持不同field embeddings的size不等
        :param dnn_units: MLP层的隐藏层size列表
        :param dropout:
        :param reduction_ratio: SENet中的缩减比率
        :param num_groups: SENet+ embedding分割的group数目
        :param bilinear_output_size: 双线性交互层的输出size
        :param bilinear_type: 双线性交互类型，['all', 'each', 'interaction']，支持其中一种
        :param dnn_activation: MLP层的激活函数
        :param dnn_use_bn: MLP层是否使用normalization
        :param dnn_l2_reg: MLP层的参数正则化
        :param bilinear_plus: 是否使用bi-linear+
        :param equal_dim: 所有field的embeddings的size是否相同
        """
        self.dnn_units = dnn_units
        self.dnn_activation = dnn_activation
        self.dnn_use_bn = dnn_use_bn
        self.dnn_l2_reg = dnn_l2_reg
        self.dropout = dropout
        self.bilinear = BiLinear(output_size=bilinear_output_size,
                                 bilinear_type=bilinear_type,
                                 bilinear_plus=bilinear_plus,
                                 equal_dim=equal_dim)
        self.senet = SENet(reduction_ratio=reduction_ratio,
                           num_groups=num_groups)

    def __call__(self, sparse_embeddings_list: List[tf.Variable],
                 dense_embeddings_list: List[tf.Variable],
                 is_training: bool = True):
        sparse_embeddings_list = [tf.layers.batch_normalization(inputs=emb, name=f'sparse_bn_{i}', training=is_training)
                                  for i, emb in enumerate(sparse_embeddings_list)]
        dense_embeddings_list = [tf.contrib.layers.layer_norm(inputs=emb,
                                                              begin_norm_axis=-1,
                                                              begin_params_axis=-1,
                                                              scope=f'dense_ln_{i}')
                                 for i, emb in enumerate(dense_embeddings_list)]
        senet_output = self.senet(sparse_embeddings_list + dense_embeddings_list)
        bilinear_output = self.bilinear(sparse_embeddings_list + dense_embeddings_list)

        output = dnn_layer(inputs=tf.concat([senet_output, bilinear_output], axis=-1),
                           is_training=is_training,
                           hidden_size=self.dnn_units,
                           activation=self.dnn_activation,
                           dropout=self.dropout,
                           use_bn=self.dnn_use_bn,
                           l2_reg=self.dnn_l2_reg)
        return output


if __name__ == '__main__':
    # test
    model = FiBiNet(dnn_units=[512, 128],
                    dropout=0.2,
                    reduction_ratio=2,
                    num_groups=2,
                    bilinear_output_size=64,
                    bilinear_type='interaction',
                    bilinear_plus=False,
                    equal_dim=False,)
    output = model([tf.placeholder(tf.float32, [None, 128]) for _ in range(20)],
                   [tf.placeholder(tf.float32, [None, 64]) for _ in range(10)],
                   is_training=True)
    print(output)
