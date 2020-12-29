
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU


class Siamese:

    def __init__(self, input_shape, embedding_shape):
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        self.branch_model = define_branch(self.input_shape, self.embedding_shape)
        self.head_model = define_head(self.embedding_shape)

        self.model = define_siamese(self.branch_model, self.head_model, self.input_shape)

    def train(self, x, y, n_epochs=100, n_batch=1):
        pairTrain, pairTest = x
        labelTrain, labelTest = y
        # train the model
        print("[INFO] training model...")
        history = self.model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
            validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
            batch_size=n_batch,
            epochs=n_epochs)

        return history

    def predict(self):
        pass

    def save_estimator(self, file):
        self.model.save(file)

    def load_trained_generator(self, file):
        self.model = load_model(file)


def define_branch(input_shape, embedding_shape=16):

    branch_input = Input(shape=input_shape)

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
    branch_output = Dense(embedding_shape)(d)

    return Model(branch_input, branch_output)


def define_head(embedding_shape=16):
    branch_a_input = Input(shape=embedding_shape)
    branch_b_input = Input(shape=embedding_shape)

    d = Concatenate()([branch_a_input, branch_b_input])
    d = Dense(1)(d)
    head_output = Activation(activation='sigmoid')(d)

    return Model([branch_a_input, branch_b_input], head_output)


def define_siamese(branch_model, head_model, input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    branch_a_output = branch_model(input_a)
    branch_b_output = branch_model(input_b)

    head_output = head_model([branch_a_output, branch_b_output])
    model = Model([input_a, input_b], head_output)

    loss = 'binary_crossentropy'
    opt = 'adam'
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
