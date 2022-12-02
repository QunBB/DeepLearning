import tensorflow as tf


def model_fn(features, labels, mode, params):  # 必须要有前面三个参数
    # feature和labels其实就是`input_fn`方法传输过来的
    # mode是用来判断你现在是训练或测试阶段
    # params是在创建`estimator`对象的输入参数
    lr = params['lr']
    try:
        init_checkpoint = params['init_checkpoint']
    except KeyError:
        init_checkpoint = None

    x = features['inputs']
    y = features['labels']

    #####################在这里定义你自己的网络模型###################
    pre = tf.layers.dense(x, 1)
    loss = tf.reduce_mean(tf.pow(pre - y, 2), name='loss')
    ######################在这里定义你自己的网络模型###################

    # 这里可以加载你的预训练模型
    assignment_map = dict()
    if init_checkpoint:
        for var in tf.train.list_variables(init_checkpoint):  # 存放checkpoint的变量名称和shape
            assignment_map[var[0]] = var[0]
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 定义你训练过程要做的事情
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # 定义你测试（验证）过程
    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'eval_loss': tf.metrics.mean_tensor(loss), "accuracy": tf.metrics.accuracy(labels, pre)}
        output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # 定义你的预测过程
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions': pre}
        output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        raise TypeError

    return output_spec


def input_fn_bulider(inputs_file, batch_size, is_training):
    name_to_features = {'inputs': tf.FixedLenFeature([3], tf.float32),
                        'labels': tf.FixedLenFeature([], tf.float32)}

    def input_fn(params):
        d = tf.data.TFRecordDataset(inputs_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle()

        # map_and_batch其实就是将map和batch结合起来而已
        d = d.apply(tf.contrib.data.map_and_batch(lambda x: tf.parse_single_example(x, name_to_features),
                                                  batch_size=batch_size))
        return d

    return input_fn


if __name__ == '__main':
    # 定义日志消息的输出级别，为了获取模型的反馈信息，选择INFO
    tf.logging.set_verbosity(tf.logging.INFO)
    # 我在这里是指定模型的保存和loss输出频率
    runConfig = tf.estimator.RunConfig(save_checkpoints_steps=1,
                                       log_step_count_steps=1)

    estimator = tf.estimator.Estimator(model_fn, model_dir='your_save_path',
                                       config=runConfig, params={'lr': 0.01})

    # log_step_count_steps控制的只是loss的global_step的输出
    # 我们还可以通过tf.train.LoggingTensorHook自定义更多的输出
    # tensor是我们要输出的内容，输入一个字典，key为打印出来的名称，value为你要输出的tensor的name
    logging_hook = tf.train.LoggingTensorHook(every_n_iter=1,
                                              tensors={'loss': 'loss:0'})

    # 其实给到estimator.train是一个dataset对象
    input_fn = input_fn_bulider('test.tfrecord', batch_size=1, is_training=True)
    estimator.train(input_fn, max_steps=1000, hooks=[logging_hook])
