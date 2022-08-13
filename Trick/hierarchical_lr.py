import tensorflow as tf

lr_dict = {'bert': 1e-5,
           'default': 1e-3}


def create_train_op(loss: tf.Tensor, global_step: tf.Tensor):
    optimizer_dict = {}
    for key in lr_dict:
        # 这里可以选择其他的优化器
        optimizer_dict[key] = tf.train.AdamOptimizer(learning_rate=lr_dict[key])

    # 这里计算梯度与学习率无关, 选择任一optimizer即可
    gradients = optimizer_dict['default'].compute_gradients(loss)

    vars_dict = {k: [] for k in lr_dict}
    for grad, var in gradients:
        layer = 'default'  # 默认归属层
        for key in lr_dict:
            if key in var.name:
                layer = key
                break
        vars_dict[layer].append((grad, var))

    train_op_list = []
    for key, var in vars_dict.items():
        # 在这里根据不同的学习率进行反向传播，更新参数
        # global_step参数None，代表global_step不变
        train_op_list.append(optimizer_dict[key].apply_gradients(vars_dict[key], global_step=None))

    # global_step在这里+1
    new_global_step = global_step + 1
    train_op_list.append(global_step.assign(new_global_step))
    train_op = tf.group(*train_op_list)

    return train_op
