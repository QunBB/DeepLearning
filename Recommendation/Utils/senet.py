import tensorflow as tf
from typing import List


class SENet:
    """
    SENet+ Layer，支持不同field embeddings的size不等
    """
    def __init__(self,
                 reduction_ratio: int,
                 num_groups: int):
        self.reduction_ratio = reduction_ratio
        self.num_groups = num_groups

    def __call__(self, embeddings_list: List[tf.Variable]):
        """

        :param embeddings_list: [embedding_1,...,embedding_i,...,embedding_f]，f为field的数目，embedding_i is [bs, dim]
        :return:
        """
        for emb in embeddings_list:
            assert len(emb.shape.as_list()) == 2, 'field embeddings must be rank 2 tensors'

        field_size = len(embeddings_list)
        feature_size_list = [emb.shape.as_list()[-1] for emb in embeddings_list]

        # Squeeze
        group_embeddings_list = [tf.reshape(emb, [-1, self.num_groups, tf.shape(emb)[-1] // self.num_groups]) for emb in embeddings_list]
        Z = [tf.reduce_mean(emb, axis=-1) for emb in group_embeddings_list] + [tf.reduce_max(emb, axis=-1) for emb in group_embeddings_list]
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


if __name__ == '__main__':
    # test
    se = SENet(2, 2)
    output = se([tf.placeholder(tf.float32, [None, 64]) for _ in range(10)] +
                [tf.placeholder(tf.float32, [None, 128]) for _ in range(20)])
    print(output)
