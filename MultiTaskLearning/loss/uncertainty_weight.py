import tensorflow as tf


def uncertainty_to_weigh_losses(loss_list):
    """
    所有task的loss列表，tensor格式
    :param loss_list:
    :return: tensor格式的综合loss
    """
    loss_n = len(loss_list)
    uncertainty_weight = [
        tf.get_variable("uncertainty_weight_"+str(i), shape=(), initializer=[1 / loss_n]
                        ) for i in range(loss_n)
    ]

    final_loss = []
    for i in range(loss_n):
        final_loss.append(tf.div(loss_list[i], 2*tf.square(uncertainty_weight[i])) + tf.log(uncertainty_weight[i]))

    return tf.add_n(final_loss)
