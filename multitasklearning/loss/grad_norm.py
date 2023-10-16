import math
import tensorflow as tf

from typing import List


def l2_normalize(x):
    return tf.sqrt(tf.reduce_sum(tf.pow(x, 2)))


def grad_norm(loss_tensor_list: List[tf.Variable],
              last_shared_weight: tf.Variable,
              optimizer: tf.train.Optimizer,
              loss_0: List,
              alpha: float = 0.12):
    """

    :param loss_tensor_list: 所有task的loss列表，tensor形式
    :param last_shared_weight: 最后一层共享层的参数，tensor形式
    :param optimizer: 优化器，tf.train.Optimizer
    :param loss_0: 所有task的初设loss列表
    :param alpha: grad_norm的超参数
    :return:
    """
    # 多任务学习的task数量
    task_n = len(loss_tensor_list)

    # 每个task的loss权重
    w = [tf.get_variable("loss_weight_" + str(i), initializer=1.) for i in range(task_n)]

    # 每个task的正则化梯度
    gradient_norm = [
        l2_normalize(tf.gradients(w[i] * loss_tensor_list[i], last_shared_weight)[0]) for i in range(task_n)
    ]

    gradient_norm_avg = tf.reduce_mean(gradient_norm)

    # 每个task的loss比率
    loss_ratio = [loss_tensor_list[i] / loss_0[i] for i in range(task_n)]

    loss_ratio_avg = tf.reduce_mean(loss_ratio)

    # 每个task的相对学习速度
    train_rate = [l / loss_ratio_avg for l in loss_ratio]

    # 正则化梯度loss
    loss_grad = [tf.abs(gradient_norm[i] - gradient_norm_avg * tf.pow(train_rate[i], alpha)) for i in range(task_n)]
    loss_grad = tf.reduce_sum(loss_grad)
    # 仅对loss权重即w_i 做梯度方向传播
    grad_op = optimizer.minimize(loss_grad, var_list=w)

    # 总loss
    total_loss = tf.reduce_sum([w[i] * loss_tensor_list[i] for i in range(task_n)])
    # loss_grad不参与网络层的参数的反向梯度更新
    trainable_weights = tf.trainable_variables()
    for v in w:
        trainable_weights.remove(v)
    train_op = optimizer.minimize(total_loss, var_list=trainable_weights - w)

    return total_loss, train_op, loss_grad, grad_op

# if __name__ == '__main__':
#     last_shared_weight = tf.get_variable("last_shared_weight", shape=[100, 200])
#     loss_tensor_list = [tf.reduce_sum(last_shared_weight * 0.01) for i in range(3)]
#     optimizer = tf.train.AdamOptimizer()
#     loss_0 = [math.log(3) for _ in range(3)]
#     grad_norm(loss_tensor_list,
#               last_shared_weight,
#               optimizer,
#               loss_0)
#     print("")
