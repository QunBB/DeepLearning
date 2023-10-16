import tensorflow as tf


class EMA:

    def __init__(self, global_step: tf.Variable,
                 decay: float = 0.999):

        ema = tf.train.ExponentialMovingAverage(decay, global_step)

        vars_list = tf.trainable_variables()

        # EMA平滑操作
        self.ema_op = ema.apply(vars_list)

        # 原参数替换EMA平滑参数
        self.ema_assign_op = [tf.assign(w, ema.average(w)) for w in vars_list]

        # 用于临时存储原来的参数
        backup = [tf.get_variable('ema_backup/' + self._get_var_name(w.name), shape=w.shape, dtype=w.dtype, trainable=False) for w in vars_list]
        self.weight_copy_op = [tf.assign(w1, w2) for w1, w2 in zip(backup, vars_list)]

        # 恢复原参数
        self.weight_restore_op = [tf.assign(w1, w2) for w1, w2 in zip(vars_list, backup)]

        self.sess = None

    def _get_var_name(self, name: str):
        if name.endswith(":0"):
            name = name[:-2]
        return name

    def register(self, sess: tf.Session):
        """无需创建shadow变量，ema.apply(vars_list)会自动创建"""
        self.sess = sess

    def update(self):
        """EMA平滑操作，更新shadow权重"""
        self.sess.run(self.ema_op)

    def apply_shadow(self):
        """使用shadow权重作为模型权重，并创建原模型权重备份"""
        self.sess.run(self.weight_copy_op)
        self.sess.run(self.ema_assign_op)

    def restore(self):
        """恢复模型权重"""
        self.sess.run(self.weight_restore_op)


def train_with_ema(train_op,
                   sess: tf.Session,
                   iterations: int,
                   valid_steps: int):
    global_step = tf.train.get_or_create_global_step()

    ema = EMA(global_step, decay=0.999)
    ema.register(sess)

    for i in range(iterations):

        # 常规的训练，更新权重
        sess.run(train_op)

        # 更新ema平滑权重
        ema.update()

        if (i + 1) % valid_steps == 0:
            # 使用ema权重
            ema.apply_shadow()

            # 验证工作
            print('do valid')

            # 保存模型工作
            print('save model')

            # 恢复原模型权重，继续正常的训练
            ema.restore()
