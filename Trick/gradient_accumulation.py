import tensorflow as tf

"""
steps_accumulate为梯度累积的步数，即累积`steps_accumulate`再进行一次反向传播更新参数
实现`steps_accumulate * bs`的大批次训练
"""


def create_train_op(loss: tf.Tensor,
                    global_step: tf.Tensor,
                    steps_accumulate: int):
    opt = tf.train.AdamOptimizer(0.01)

    tvs = tf.trainable_variables()

    # 创建梯度变量副本，用于累积梯度
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    # 清空梯度变量副本
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    # 计算当前批次梯度
    gvs = opt.compute_gradients(loss / steps_accumulate, tvs)

    # 将当前批次的梯度累加到`accum_vars`
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    # 使用累积的梯度，进行反向传播更新参数
    train_op = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)],
                                   global_step=global_step)

    return train_op, accum_ops, zero_ops


def train(loss: tf.Tensor, steps_accumulate: int):
    global_step = tf.train.get_or_create_global_step()
    train_op, accum_ops, zero_ops = create_train_op(loss, global_step, steps_accumulate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            # 这里是模拟使用tf.data.Dataset定义输入流
            # 如果是使用placeholder的方式，则需喂入feed_dict数据
            sess.run(accum_ops)

            if (i + 1) % steps_accumulate == 0:
                sess.run(train_op)

                sess.run(zero_ops)
