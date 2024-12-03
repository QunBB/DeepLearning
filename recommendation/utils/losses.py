import tensorflow as tf


def ranking_loss(y_true, y_pred):
    """Only compute pairs loss of positive v.s. negative samples.
    """

    z_ij = tf.reshape(y_pred, [-1, 1]) - tf.reshape(y_pred, [1, -1])

    y_true = tf.convert_to_tensor(y_true, dtype="int64")
    mask = tf.logical_and(tf.equal(tf.reshape(y_true, [-1, 1]), 1),
                          tf.equal(tf.reshape(y_true, [1, -1]), 0))
    mask = tf.cast(mask, z_ij.dtype)

    per_pair_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(z_ij, z_ij.dtype), z_ij,
                                                    reduction=tf.losses.Reduction.NONE)

    num_pairs = tf.reduce_sum(mask)

    return tf.reduce_sum(per_pair_loss * mask) / (num_pairs + 1e-7)
