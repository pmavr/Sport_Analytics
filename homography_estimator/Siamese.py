import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.initializers import RandomUniform


class Siamese:

    def __init__(self, input_shape, embedding_size=16):
        self.input_shape = input_shape
        self.embedding_size = embedding_size

        self.branch_model = self._define_branch()
        self.head_model = self._define_head(self.branch_model.output_shape)

        self.model = self._define_siamese()

    def _define_branch(self):
        branch_input = Input(shape=self.input_shape)

        d = Conv2D(4, kernel_size=7, strides=2, padding='same')(branch_input)
        d = LeakyReLU(alpha=0.1)(d)

        d = Conv2D(8, kernel_size=5, strides=2, padding='same')(d)
        d = ReLU()(d)

        d = Conv2D(16, kernel_size=3, strides=2, padding='same')(d)
        d = ReLU()(d)

        d = Conv2D(32, kernel_size=3, strides=2, padding='same')(d)
        d = ReLU()(d)

        d = Conv2D(16, kernel_size=3, strides=2, padding='same')(d)
        d = ReLU()(d)
        d = Flatten()(d)
        branch_output = Dense(self.embedding_size)(d)

        model = Model(branch_input, branch_output)

        return model

    def _define_head(self, embedding_shape):
        branch_a_output = Input(shape=embedding_shape)
        branch_b_output = Input(shape=embedding_shape)

        # head_output = tf.stack([branch_a_output, branch_b_output])
        head_output = tf.keras.layers.Concatenate()([branch_a_output, branch_b_output])

        model = Model([branch_a_output, branch_b_output], head_output)

        return model

    def _define_siamese(self):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        input_a_encoded = self.branch_model(input_a)
        input_b_encoded = self.branch_model(input_b)

        head_output = self.head_model([input_a_encoded, input_b_encoded])

        model = Model([input_a, input_b], head_output)

        return model


if __name__ == '__main__':
    import sys
    from ContrastiveLoss import ContrastiveLoss

    siamese = Siamese(input_shape=(180, 320, 1))
    criterion = ContrastiveLoss(margin=1.0)

    import numpy as np
    np.random.seed(0)
    N = 3
    x1 = tf.constant(np.random.rand(N, 180, 320, 1), dtype=tf.float32)
    x2 = tf.constant(np.random.rand(N, 180, 320, 1), dtype=tf.float32)

    y1 = tf.constant(np.random.rand(N, 1), dtype=tf.float32)
    y_zeros = tf.zeros((N, 1))
    y_ones = tf.ones((N, 1))

    y_true = tf.where((y1 > 0), y_ones, y_zeros)

    y_pred = siamese.model((x1, x2))
    loss = criterion(y_true, y_pred)
    f1, f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)
    print(f1)
    print(f2)
    print(loss)
    sys.exit()
