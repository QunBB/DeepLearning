"""
《CowClip: Reducing CTR Prediction Model Training Time from 12 hours to 10 minutes on 1 GPU》
论文地址：https://arxiv.org/abs/2204.06240
开源地址：https://github.com/bytedance/LargeBatchCTR
"""
import tensorflow as tf

import os

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def cow_clip(w, g, ratio=1, ids=None, pos=None, cnts=None, min_w=0.03):
    """

    :param w: embedding变量
    :param g: embedding变量的梯度
    :param ratio: 论文中的r
    :param ids: 去重的ID
    :param pos: tf.unique(pos).y 为去重后的ID位置索引
    :param cnts: ID的个数统计
    :param min_w: 论文中的 \zeta
    :return:
    """
    values = tf.gather(g.values, tf.unique(pos).y)
    clipnorm = tf.norm(tf.gather(w, ids), axis=-1)

    # 与论文一致，clipnorm先乘ratio，再与min_w取max
    # 但原作者实现是 clipnorm先与min_w取max，再乘ratio
    clipnorm = tf.maximum(ratio * clipnorm, min_w)

    clip_t = clipnorm * tf.cast(cnts, tf.float32)

    l2sum_row = tf.reduce_sum(values * values, axis=-1)
    pred = l2sum_row > 0
    l2sum_row_safe = tf.where(pred, l2sum_row, tf.ones_like(l2sum_row))
    l2norm_row = tf.sqrt(l2sum_row_safe)
    intermediate = values * tf.expand_dims(clip_t, -1)
    g_clip = intermediate / tf.expand_dims(tf.maximum(l2norm_row, clip_t), -1)

    # tensorflow中出现多次的ID的梯度是重复存储，所以这里进行复原
    g_clip = tf.repeat(g_clip, cnts, axis=0)
    indices = tf.repeat(ids, cnts)

    return tf.IndexedSlices(g_clip, indices, g.dense_shape)


if __name__ == '__main__':
    import numpy as np

    inputs_id = tf.placeholder(tf.int32, [None, 20])
    labels = tf.placeholder(tf.int32, [None])
    embedding_table = tf.get_variable('embedding_table', shape=[1000, 128])
    test_weight = tf.get_variable('test_weight', shape=[1, 128])

    # 存储ID embedding变量及对应的ID输入
    ids_variables_dict = {embedding_table.name: inputs_id}

    dnn_input = tf.reduce_max(tf.nn.embedding_lookup(embedding_table, inputs_id), axis=1)
    predictions = tf.layers.dense(dnn_input + test_weight, 1)
    predictions = tf.reshape(predictions, [-1])
    predictions = tf.nn.sigmoid(predictions)

    loss = tf.reduce_mean(tf.pow(tf.cast(labels, tf.float32) - predictions, 2))

    embedding_grad = []
    embedding_var = []
    dense_grad = []
    dense_var = []
    for var, grad in zip(tf.trainable_variables(), tf.gradients(loss, tf.trainable_variables())):
        print(var, grad)
        # 如果是经过embedding_lookup的ID类embedding，`梯度类型为IndexedSlices，其他为tensor
        if isinstance(grad, tf.IndexedSlices) and var.name in ids_variables_dict:
            unique_ids, pos, ids_count = tf.unique_with_counts(tf.reshape(ids_variables_dict[var.name], [-1]))
            clip_grad = cow_clip(var, grad, pos=pos, ids=unique_ids, cnts=ids_count)
            embedding_grad.append(clip_grad)
            embedding_var.append(var)
        else:
            dense_grad.append(grad)
            dense_var.append(var)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.apply_gradients(zip(embedding_grad + dense_grad, embedding_var + dense_var))

    ######################## 测试程序 ########################
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # feed_dict = {
    #     # inputs_id: np.random.randint(0, 20, [1, 20]),
    #     inputs_id: [[14, 9, 5, 1, 3, 12, 14, 11, 12, 6, 1, 16, 13, 6, 16, 19, 13, 0, 3, 19]],
    #     labels: np.random.randint(0, 10, [10])
    # }
    #
    # # print(sess.run(embedding_grad, feed_dict=feed_dict))
    # print(sess.run(tf.unique_with_counts(tf.reshape(inputs_id, [-1])), feed_dict=feed_dict))
    # print(sess.run(inputs_id, feed_dict=feed_dict))
    # grad_v, grad_i = sess.run([embedding_grad[0].values, embedding_grad[0].indices], feed_dict=feed_dict)
    # print(grad_v.shape, grad_v[0], grad_v[6], grad_v[1])
    # print(grad_i.shape, grad_i)
    #
    # for _ in range(1000):
    #     _, _loss = sess.run([train_op, loss],
    #                         feed_dict={
    #                             inputs_id: np.random.randint(0, 20, [32, 20]),
    #                             labels: np.random.randint(0, 1, [32])
    #                         }
    #                         )
    #     print(_loss)
