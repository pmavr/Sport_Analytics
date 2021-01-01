
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU, ReLU


class Siamese:

    def __init__(self, input_shape, embedding_shape=16):
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        self.branch_model = self._define_branch()
        self.head_model = self._define_head()

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
        branch_output = Dense(self.embedding_shape)(d)

        return Model(branch_input, branch_output)

    def _define_head(self):
        branch_a_output = Input(shape=self.embedding_shape)
        branch_b_output = Input(shape=self.embedding_shape)

        head_output = Concatenate()([branch_a_output, branch_b_output])

        return Model([branch_a_output, branch_b_output], head_output)

    def _define_siamese(self):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        branch_a_output = self.branch_model(input_a)
        branch_b_output = self.branch_model(input_b)

        head_output = self.head_model([branch_a_output, branch_b_output])
        model = Model([input_a, input_b], head_output)

        return model

    def _pair_generator(self, x_train, y_train, batch_size):
        return None


if __name__ == '__main__':
    import sys
    from tensorflow.keras.optimizers import Adam

    model_path = utils.get_homography_estimator_model_path()

    img_shape = (180, 320, 1)
    learning_rate = .01
    batch_size = 64
    num_epoch = 10

    optimizer = Adam(lr=learning_rate, weight_decay=0.000001)

    siamese = Siamese(input_shape=img_shape)

    siamese.model.compile(loss='contrastive_loss', optimizer=optimizer, metrics=['accuracy'])



    siamese.model.fit(train_generator,
                   steps_per_epoch=train_steps,
                   epochs=num_epoch,
                   validation_data=test_generator,
                   validation_steps=test_steps
                   )

    sys.exit()
