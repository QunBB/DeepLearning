import tensorflow as tf
from typing import Optional, Union, List


class BaseLoss(object):
    """Inherit from this class when implementing new losses."""

    def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
        raise NotImplementedError()


class LabelSmoothingCrossEntropy(BaseLoss):
    def __init__(self,
                 smoothing: Optional[float] = 0.1,
                 with_logits: Optional[bool] = True):
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.with_logits = with_logits

    def calculate_loss(self, inputs: tf.Tensor, target: tf.Tensor, **unused_params) -> tf.Tensor:
        if self.with_logits:
            logprobs = tf.nn.log_softmax(inputs, axis=-1)
        else:
            logprobs = tf.log(inputs)

        nll_loss = -tf.gather_nd(logprobs,
                                 indices=tf.stack([tf.range(tf.shape(target)[0]), target], axis=1))
        smooth_loss = -tf.reduce_mean(logprobs, axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return tf.reduce_mean(loss)


class FocalLoss(BaseLoss):

    def __init__(self,
                 alpha: Union[List[float], float],
                 gamma: Optional[int] = 2,
                 with_logits: Optional[bool] = True):
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-8
        self.with_logits = with_logits

    def _binary_class(self, inputs):
        probs = tf.nn.sigmoid(inputs) if self.with_logits else inputs
        probs += self.smooth
        logprobs = tf.math.log(probs)

        loss = self.alpha * tf.pow(1 - probs, self.gamma) * logprobs

        return tf.negative(loss)

    def _multiple_class(self, inputs, target):
        probs = tf.nn.softmax(inputs, axis=-1) if self.with_logits else inputs
        probs += self.smooth
        logprobs = tf.math.log(probs)

        target = tf.cast(tf.one_hot(target, tf.shape(inputs)[-1]), inputs.dtype)

        # alpha = tf.reshape(self.alpha, [1] + [-1])
        loss = self.alpha * target * tf.pow(1 - probs, self.gamma) * logprobs
        loss = tf.reduce_sum(tf.negative(loss), axis=-1)

        return loss

    def calculate_loss(self, inputs: tf.Tensor, target: tf.Tensor, **unused_params) -> tf.Tensor:
        shape = inputs.shape
        if not isinstance(shape, (list, tuple)):
            shape = shape.as_list()

        if len(shape) > 1 and shape[-1] != 1:
            loss = self._multiple_class(inputs, target)
        else:
            loss = self._binary_class(inputs)

        return tf.reduce_mean(loss)


class DiceLoss(BaseLoss):
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 alpha: Optional[float] = 0.0,
                 set_level: Optional[bool] = True):
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.with_logits = with_logits
        self.alpha = alpha
        self.set_level = set_level

    def _compute_dice_loss(self, inputs, target, axis=-1):
        if self.set_level:
            axis = 0

        inputs = ((1 - inputs) ** self.alpha) * inputs
        interection = tf.reduce_sum(inputs * target, axis=axis)

        if self.square_denominator:
            inputs = tf.square(inputs)
            target = tf.square(target)

        loss = 1 - ((2 * interection + self.smooth) /
                    (tf.reduce_sum(inputs, axis=axis) + tf.reduce_sum(target, axis=axis) + self.smooth))

        return loss

    def calculate_loss(self, inputs, target, **unused_params) -> tf.Tensor:
        shape = inputs.shape
        if not isinstance(shape, (list, tuple)):
            shape = shape.as_list()

        if len(shape) > 1 and shape[-1] != 1:  # 多分类
            if self.with_logits:
                inputs = tf.nn.softmax(inputs, axis=-1)
            target = tf.cast(tf.one_hot(target, tf.shape(inputs)[-1]), inputs.dtype)

            if not self.set_level:
                inputs = tf.expand_dims(inputs, axis=1)
                target = tf.expand_dims(target, axis=1)

                loss = self._compute_dice_loss(inputs, target, axis=1)
                loss = tf.reduce_sum(loss, axis=1)  # 多个分类标签的loss相加
            else:
                loss = self._compute_dice_loss(inputs, target, axis=1)
        else:  # 二分类
            if self.set_level:
                inputs = tf.reshape(inputs, [-1])
            else:
                inputs = tf.reshape(inputs, [-1, 1])
                target = tf.reshape(target, [-1, 1])

            if self.with_logits:
                inputs = tf.nn.sigmoid(inputs)

            loss = self._compute_dice_loss(inputs, tf.cast(target, inputs.dtype))

        if self.set_level:
            # set-level下，此处的dice loss为多个分类标签的loss，因此只能相加
            return tf.reduce_sum(loss)
        else:
            # 非set-level下，此处的dice loss为每个样本的loss
            return tf.reduce_mean(loss)


if __name__ == '__main__':
    # for test
    import numpy as np
    np.random.seed(2022)

    multi_pred, multi_target = np.random.random([32, 3]), np.random.randint(0, 3, [32])
    binary_pred, binary_target = np.random.random([32]), np.random.randint(0, 2, [32])

    sess = tf.Session()

    print('*'*20, 'Label Smoothing', '*'*20)
    print(sess.run(LabelSmoothingCrossEntropy().calculate_loss(multi_pred, multi_target)))

    focal = FocalLoss(alpha=[0.25, 0.25, 0.5])
    print('*'*20, 'Focal multi class', '*'*20)
    print(sess.run(focal.calculate_loss(multi_pred, multi_target)))
    focal = FocalLoss(alpha=0.25)
    print('*'*20, 'Focal binary class', '*'*20)
    print(sess.run(focal.calculate_loss(binary_pred, binary_target)))

    dice = DiceLoss(square_denominator=True, set_level=True)
    print('*'*20, 'Dice multi class', '*'*20)
    print(sess.run(dice.calculate_loss(multi_pred, multi_target)))

    print('*'*20, 'Dice binary class', '*'*20)
    print(sess.run(dice.calculate_loss(binary_pred, binary_target)))

