import sys
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils


class ContrastiveLoss(Loss):

    def __init__(self,
                 margin=1.0,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='contrastive_loss'):
        super(ContrastiveLoss, self).__init__(reduction=reduction, name=name)
        self.margin = margin

    def __call__(self, y_true, y_pred, **kwargs):
        """
        :param margin:
        :param x1: N * D
        :param x2: N * D
        :param y_true: 0 for un-similar, 1 for similar
        :return:
        """
        # x1, x2 = tf.unstack(y_pred, axis=0)
        x1, x2 = tf.split(y_pred, num_or_size_splits=2, axis=1)
        y_true = tf.squeeze(y_true, axis=-1)
        assert len(x1.shape) == 2
        assert len(x2.shape) == 2
        assert len(y_true.shape) == 1
        assert y_true.shape[0] == x1.shape[0]

        distance = tf.norm(tf.subtract(x1, x2), ord=2, axis=1)
        loss = y_true * tf.pow(distance, 2) + (1 - y_true) * tf.pow(tf.maximum(self.margin - distance, 0.), 2)
        return .5 * tf.reduce_sum(loss)

if __name__ == '__main__':
    contrastive_loss = ContrastiveLoss(margin=1.0)
    import numpy as np
    np.random.seed(0)
    N = 10
    branch_a_output = tf.constant(np.random.rand(N, 16), dtype=tf.float32)
    branch_b_output = tf.constant(np.random.rand(N, 16), dtype=tf.float32)
    # y_pred = tf.stack([branch_a_output, branch_b_output])
    y_pred = tf.keras.layers.Concatenate()([branch_a_output, branch_b_output])

    y1 = tf.constant(np.random.rand(N, 1))
    y_zeros = tf.zeros((N, 1))
    y_ones = tf.ones((N, 1))

    y_true = tf.where((y1 > 0), y_ones, y_zeros)
    # y_true = tf.squeeze(y_true)
    # print(label.shape)
    loss = contrastive_loss(y_true, y_pred)
    print(loss.shape)
    print(loss)

    sys.exit()
