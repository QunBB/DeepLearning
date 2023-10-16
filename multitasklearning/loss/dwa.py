import math

T = 20


def dynamic_weight_average(loss_t_1, loss_t_2):
    """

    :param loss_t_1: 每个task上一轮的loss列表，并且为标量
    :param loss_t_2:
    :return:
    """
    # 第1和2轮，w初设化为1，lambda也对应为1
    if not loss_t_1 or not loss_t_2:
        return 1

    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)

    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    return [task_n * l / lamb_sum for l in lamb]
