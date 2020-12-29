import numpy as np
import tensorflow as tf
from tensorflow_addons.losses.metric_learning import pairwise_distance

class ContrastiveLoss:
    def __init__(self, margin):
        """
        :param margin: 1.0
        """
        self.margin = margin

    def forward(self, x1, x2, label):
        """
        :param x1: N * D
        :param x2: N * D
        :param label: 0 for un-similar, 1 for similar
        :return:
        """
        assert len(x1.shape) == 2
        assert len(x2.shape) == 2
        assert len(label.shape) == 1
        assert label.shape[0] == x1.shape[0]

        # pdist = nn.PairwiseDistance(p=2)
        # dist = pdist(x1, x2)
        dist = tf.norm(x1 - x2, ord=2, axis=1)
        loss = label * tf.pow(dist, 2) + (1-label) \
               * tf.pow(tf.clip_by_value(
                                    self.margin - dist,
                                    clip_value_min=0.0,
                                    clip_value_max=np.inf), 2)
        loss = tf.reduce_sum(loss)

        return 0.5 * loss


if __name__ == '__main__':
    closs = ContrastiveLoss(margin=1.0)
    N = 2
    x1 = tf.constant([[-0.2600, 0.0971],
                      [1.5916, 1.2401]], dtype=tf.float32)

    x2 = tf.constant([[-0.6877, -0.6241],
                      [-0.3784, -1.4103]], dtype=tf.float32)

    label = tf.constant([[1.], [0.]])
    label = tf.squeeze(label)
    # print(label.shape)
    loss = closs.forward(x1, x2, label)
    print(loss.shape)
    print(loss)