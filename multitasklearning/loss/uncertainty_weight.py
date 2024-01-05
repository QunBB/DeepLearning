import tensorflow as tf


def uncertainty_to_weigh_losses(loss_list):
    """
    所有task的loss列表，tensor格式
    :param loss_list:
    :return: tensor格式的综合loss
    """
    loss_n = len(loss_list)
    # 初始化`log(uncertainty_weight)`变量，来代替`uncertainty_weight`，可以避免后续计算`log(uncertainty_weight)`，出现Nan的问题
    # 这里初始化的变量是乘以2的即`log(uncertainty_weight)*2`，来代替后续的平方计算
    uncertainty_weight_log = [
        tf.get_variable("uncertainty_weight_log_"+str(i), shape=(), initializer=[0.]
                        ) for i in range(loss_n)
    ]

    final_loss = []
    for i in range(loss_n):
        # log等式替换
        final_loss.append(tf.div(loss_list[i], 2*tf.exp(uncertainty_weight_log[i])) + uncertainty_weight_log[i] / 2)

    return tf.add_n(final_loss)
