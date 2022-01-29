import tensorflow as tf
from typing import List


class MIND:

    def __init__(self, k_max: int,
                 p: float,
                 dnn_hidden_size: List[int],
                 dnn_activation,
                 dropout: float,
                 use_bn: bool,
                 l2_reg: float,
                 num_sampled: int,
                 dim: int,
                 user_vocab_size: List[int],
                 item_vocab_size: List[int],
                 all_item_idx: List[List[int]]):
        """

        :param k_max: 兴趣向量的个数K
        :param p: 调整attention分布的参数
        :param dnn_hidden_size: 如 [256, 128]
        :param dnn_activation: 如 tf.nn.relu
        :param dropout:
        :param use_bn: 是否使用batch_normalization
        :param l2_reg: l2正则化惩罚项
        :param num_sampled: 随机负采样的个数，问题即转化为对应 num_sampled 分类
        :param dim: 用户向量和item向量的维度dim
        :param user_vocab_size: 所有用户属性的特征个数列表，例如有1w用户id、3种性别、10个年龄段，则输入[10000, 3, 10]
        :param item_vocab_size: 所有用户属性的特征个数列表，理解同上
        :param all_item_idx: 所有item的特征输入
        """
        self.k_max = k_max
        self.p = p
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_activation = dnn_activation
        self.dropout = dropout
        self.use_bn = use_bn
        self.l2_reg = l2_reg
        self.num_sampled = num_sampled

        self.user_tables = [tf.get_variable("user_table_" + str(i), shape=[size, dim]
                                            ) for i, size in enumerate(user_vocab_size)]
        self.item_tables = [tf.get_variable("item_table_" + str(i), shape=[size, dim]
                                            ) for i, size in enumerate(item_vocab_size)]

        self.item_vocab_size = len(all_item_idx[0])

        with tf.variable_scope("embedding_layer"):
            self.all_item_embedding = sum([tf.nn.embedding_lookup(item_table, item_ids) for item_table, item_ids in
                                           zip(self.item_tables, all_item_idx)]) / len(all_item_idx)

        self.zero_bias = tf.get_variable("zero_bias", shape=[self.item_vocab_size], trainable=False)

    def get_item_embedding(self):
        """
        获取所有item的向量
        :return:
        """
        return self.all_item_embedding

    def __call__(self, user_feat_inputs: List,
                 behavior_item_inputs: List,
                 target_item_inputs: List,
                 seq_len: tf.Variable,
                 training: bool):
        """

        :param user_feat_inputs: 用户特征输入列表，特征顺序要与上述的user_vocab_size一致，[ tensor[batch_size], .....]
        :param behavior_item_inputs: 用户的历史行为item输入列表, [ tensor[batch_size, max_seq_len], .....]
        :param target_item_inputs: 目标item输入列表, [ tensor[batch_size], .....]
        :param seq_len: 用户的历史行为item长度, tensor[batch_size]
        :param training:
        :return:
        """
        with tf.variable_scope("embedding_layer"):
            behavior_item_embedding = [tf.nn.embedding_lookup(table, inputs) for table, inputs in zip(self.item_tables, behavior_item_inputs)]
            # [batch_size, max_seq_len, dim]
            behavior_item_embedding = sum(behavior_item_embedding) / len(behavior_item_embedding)

            user_embedding = [tf.nn.embedding_lookup(table, inputs) for table, inputs in zip(self.user_tables, user_feat_inputs)]
            # [batch_size, dim*n]
            user_embedding = tf.concat(user_embedding, axis=-1)

        with tf.variable_scope("multi_interest_extractor_layer"):
            # [batch_size, k_max, dim]
            interest_capsules = self.capsule_network(behavior_item_embedding, self.k_max, seq_len)

        with tf.variable_scope("relu_dnn_layer"):
            user_embedding = tf.expand_dims(user_embedding, axis=1)
            user_embedding = tf.tile(user_embedding, [1, self.k_max, 1])

            dnn_inputs = tf.concat([interest_capsules, user_embedding], axis=-1)

            user_vectors = self.dnn_layer(dnn_inputs, hidden_size=self.dnn_hidden_size,
                                          activation=self.dnn_activation,
                                          dropout=self.dropout,
                                          use_bn=self.use_bn,
                                          l2_reg=self.l2_reg)
        if not training:
            return user_vectors

        with tf.variable_scope("embedding_layer"):
            target_item_embedding = [tf.nn.embedding_lookup(table, inputs) for table, inputs in zip(self.item_tables, target_item_inputs)]
            target_item_embedding = sum(target_item_embedding) / len(target_item_embedding)

        with tf.variable_scope("label_aware_attention_layer"):
            user_vectors, loss = self.label_aware_attention_layer(user_vectors, target_item_embedding, seq_len)

        return user_vectors, loss

    def label_aware_attention_layer(self, user_vectors, target_item_embedding, seq_len):
        attention_weight = tf.pow(tf.matmul(user_vectors, tf.expand_dims(target_item_embedding, axis=-1)), self.p)

        # Dynamic interest number
        # 根据用户历史行为长度，动态计算K
        # 其实原理就是让动态算出K之后，让后面的k_max - K个不参与计算即可
        k_user = tf.cast(tf.maximum(
            1.,
            tf.minimum(
                tf.cast(self.k_max, dtype="float32"),
                tf.log1p(tf.cast(seq_len, dtype="float32")) / tf.log(2.)
            )
        ), dtype="int64")
        seq_mask = tf.expand_dims(tf.sequence_mask(k_user, self.k_max), axis=-1)
        padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
        attention_weight = tf.where(seq_mask, attention_weight, padding)

        # 由于后面k_max - K个被替换为很小的负数，softmax后就会变为0
        attention_weight = tf.nn.softmax(attention_weight)

        user_vectors = tf.reduce_sum(user_vectors * attention_weight, axis=1)

        # 随机负采样
        loss = tf.nn.sampled_softmax_loss(weights=self.all_item_embedding,
                                          biases=self.zero_bias,
                                          labels=tf.reshape(tf.constant(list(range(self.item_vocab_size))), [-1, 1]),
                                          inputs=user_vectors,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.item_vocab_size,
                                          )
        loss = tf.reduce_mean(loss)

        return user_vectors, loss

    def capsule_network(self, low_capsule, k_max, seq_len):
        max_seq_len = low_capsule.shape.as_list()[-2]
        dim = low_capsule.shape.as_list()[-1]

        bilinear_mapping_matrix = tf.get_variable(name='bilinear_mapping_matrix',
                                                  # shape=[dim, dim],
                                                  initializer=tf.truncated_normal(shape=[dim, dim]))

        # 高斯分布初设化
        routing_logits = tf.get_variable(name='routing_logits',
                                         initializer=tf.truncated_normal(shape=[1, k_max, max_seq_len]),
                                         trainable=False)
        # 用户行为不足N个的需要mask
        seq_len_tile = tf.tile(tf.expand_dims(seq_len, axis=-1), [1, k_max])
        mask = tf.sequence_mask(seq_len_tile, max_seq_len)
        pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
        routing_logits_with_padding = tf.where(mask, routing_logits, pad)

        for _ in range(3):
            w = tf.nn.softmax(routing_logits_with_padding)

            low_capsule_mapping = tf.tensordot(low_capsule, bilinear_mapping_matrix, axes=1)
            candidate_vector = tf.matmul(w, low_capsule_mapping)

            high_capsule = self.squash(candidate_vector)

            delta_routing_logits = tf.reduce_sum(tf.matmul(high_capsule, low_capsule_mapping, transpose_b=True),
                                                 axis=0, keepdims=True)

            routing_logits = tf.assign_add(routing_logits, delta_routing_logits)

        return high_capsule

    def squash(self, z):
        z_l2 = tf.reduce_sum(tf.square(z), axis=-1, keep_dims=True)
        z_l1 = tf.sqrt(z_l2 + 1e-8)

        return z / z_l1 * z_l2 / (1 + z_l2)

    def dnn_layer(self, inputs, hidden_size, activation, dropout, use_bn, l2_reg):
        output = inputs
        for size in hidden_size:
            output = tf.layers.dense(output, size,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                     kernel_initializer=tf.glorot_normal_initializer())
            if use_bn:
                output = tf.layers.batch_normalization(output)

            output = activation(output)

            output = tf.nn.dropout(output, 1 - dropout)

        return output


if __name__ == '__main__':
    import random

    model = MIND(k_max=3,
                 p=10,
                 dnn_hidden_size=[256, 128],
                 dnn_activation=tf.nn.relu,
                 dropout=0.1,
                 use_bn=True,
                 l2_reg=0.00001,
                 num_sampled=20,
                 dim=128,
                 user_vocab_size=[10000, 3, 10],
                 item_vocab_size=[10000, 100],  # 1w个item_id，100个cat_id
                 all_item_idx=[[i for i in range(10000)], [random.randint(0, 100) for _ in range(10000)]]  # 这里就对应输入1w个item的item_id和cat_id
                 )
    # batch输入
    user_feat_inputs = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(3)]
    behavior_item_inputs = [tf.placeholder(dtype=tf.int32, shape=[None, 20]) for _ in range(2)]
    target_item_inputs = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(2)]
    seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
    training = True
    user_vectors, loss = model(user_feat_inputs,
                               behavior_item_inputs,
                               target_item_inputs,
                               seq_len,
                               training)
    item_embeddings = model.get_item_embedding()
    print(user_vectors)
    print(loss)
    print(item_embeddings)
