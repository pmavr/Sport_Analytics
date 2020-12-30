import sys
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class ContrastiveLoss(tf.keras.losses.Loss):

    def __init__(self,
                 embedding_shape=16,
                 margin=1.0,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='contrastive_loss'):
        super(ContrastiveLoss, self).__init__(reduction=reduction, name=name)
        self.embedding_shape = embedding_shape
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        :param margin:
        :param x1: N * D
        :param x2: N * D
        :param y_true: 0 for un-similar, 1 for similar
        :return:
        """
        x1, x2 = y_pred
        assert len(x1.shape) == self.embedding_shape
        assert len(x2.shape) == self.embedding_shape
        assert len(y_true.shape) == 1
        assert y_true.shape[0] == x1.shape[0]

        distance = tf.norm(tf.subtract(x1, x2), ord=2, axis=1)
        return y_true * tf.pow(distance, 2) + (1 - y_true) * tf.pow(tf.maximum(self.margin - distance, 0.), 2)


if __name__ == '__main__':
    contrastive_loss = ContrastiveLoss(embedding_shape=2, margin=1.0)
    branch_1_output = tf.constant([[-50.2600, 50.0971], [1.5916, 1.2401]], dtype=tf.float32)
    branch_2_output = tf.constant([[-0.2677, 0.0941], [1.5784, 1.2403]], dtype=tf.float32)
    y_pred = [branch_1_output, branch_2_output]
    y_true = tf.constant([[1.], [1.]])
    y_true = tf.squeeze(y_true)
    # print(label.shape)
    loss = contrastive_loss(y_true, y_pred)
    print(loss.shape)
    print(loss)

    sys.exit()
