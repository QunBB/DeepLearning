import tensorflow as tf

from estimator import model_fn, input_fn_bulider

# 设置训练多少步就进行模型的保存
runConfig = tf.estimator.RunConfig(save_checkpoints_steps=10)

estimator = tf.estimator.Estimator(model_fn,
                                   model_dir='your_save_path',
                                   config=runConfig,
                                   params={'lr': 0.01})

# 在这里定义一个early stop
# 在eval过程执行early stop判断，所以评判标准也是eval数据集的metric_name
# max_steps_without_decrease：loss最多多少次不降低就停止。进行一次eval相当于一步。
early_stop = tf.estimator.experimental.stop_if_no_decrease_hook(estimator,
                                                                metric_name='loss',
                                                                max_steps_without_decrease=1,
                                                                run_every_steps=1,
                                                                run_every_secs=None)

logging_hook = tf.train.LoggingTensorHook(every_n_iter=1,
                                          tensors={'loss': 'loss:0'})

# 定义训练(train)过程的数据输入方式
train_input_fn = input_fn_bulider('train.tfrecord', batch_size=1, is_training=True)
# 定义验证(eval)过程的数据输入方式
eval_input_fn = input_fn_bulider('eval.tfrecord', batch_size=1, is_training=False)

# 创建一个TrainSpec实例
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=100,
                                    hooks=[logging_hook, early_stop])
# 创建一个EvalSpec实例
eval_spec = tf.estimator.EvalSpec(eval_input_fn)

# 流程：训练train --> 验证eval --> 判断是否要early stop --> 保存模型 --> 训练train
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

