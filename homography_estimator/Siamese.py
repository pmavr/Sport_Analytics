
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate
from tensorflow.keras.layers import LeakyReLU, ReLU


class Branch(Model):

    def __init__(self, input_shape, output_shape):
        super(Branch, self).__init__()

        self.input_shape_ = input_shape
        self.embedding_size = output_shape

        self.layers_ = []
        self.first_layer = Conv2D(4, kernel_size=7, strides=2, padding='same', input_shape=self.input_shape_)
        self.layers_.append(LeakyReLU(alpha=0.1))
        self.layers_.append(Conv2D(8, kernel_size=5, strides=2, padding='same'))
        self.layers_.append(ReLU())
        self.layers_.append(Conv2D(16, kernel_size=3, strides=2, padding='same'))
        self.layers_.append(ReLU())
        self.layers_.append(Conv2D(32, kernel_size=3, strides=2, padding='same'))
        self.layers_.append(ReLU())
        self.layers_.append(Conv2D(16, kernel_size=3, strides=2, padding='same'))
        self.layers_.append(ReLU())
        self.layers_.append(Flatten())
        self.layers_.append(Dense(self.embedding_size))

    def call(self, inputs, training=None, mask=None):
        x = self.first_layer(inputs)
        for layer in self.layers_:
            x = layer(x)
        return x

    def get_config(self):
        pass


class Siamese(Model):

    def __init__(self, input_shape=(180, 320, 1), embedding_size=16):
        super(Siamese, self).__init__()
        self.branch_model = Branch(input_shape, embedding_size)

    def _forward_one_branch(self, x):
        x = self.branch_model(x)
        x = tf.linalg.normalize(x, ord='euclidean', axis=1)[0]
        return x

    def call(self, inputs, training=None, mask=None):
        inp1, inp2 = inputs
        x1 = self._forward_one_branch(inp1)
        x2 = self._forward_one_branch(inp2)
        return Concatenate()([x1, x2])

    def get_config(self):
        pass

if __name__ == '__main__':
    import sys
    from ContrastiveLoss import ContrastiveLoss

    siamese = Siamese(input_shape=(180, 320, 1))
    criterion = ContrastiveLoss(margin=1.0)

    import numpy as np
    np.random.seed(0)
    N = 2
    x1 = tf.constant(np.random.rand(N, 180, 320, 1), dtype=tf.float32)
    x2 = tf.constant(np.random.rand(N, 180, 320, 1), dtype=tf.float32)

    y1 = tf.constant(np.random.rand(N, 1), dtype=tf.float32)
    y_zeros = tf.zeros((N, 1))
    y_ones = tf.ones((N, 1))

    y_true = tf.where((y1 > 0), y_ones, y_zeros)

    y_pred = siamese((x1, x2))
    loss = criterion(y_true, y_pred)
    # f1, f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)
    print(y_pred[0])
    print(y_pred[1])
    print(loss)
    sys.exit()
