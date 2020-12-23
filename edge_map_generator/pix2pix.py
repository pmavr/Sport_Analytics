import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

from data_preparation.prepare_data import prepare_input
import utils


class Pix2Pix:

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.discriminator = define_discriminator(self.input_shape)
        self.generator = define_generator(self.input_shape)
        self.gan = define_gan(self.generator, self.discriminator, self.input_shape)

    def train(self, x, y, n_epochs=100, n_batch=1):
        dataset = x, y
        n_patch = self.discriminator.output_shape[1]
        train_a, train_b = dataset
        bat_per_epo = int(len(train_a) / n_batch)
        n_steps = bat_per_epo * n_epochs

        for i in range(n_steps):
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            x_fake_b, y_fake = generate_fake_samples(self.generator, X_realA, n_patch)
            d_loss1 = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = self.discriminator.train_on_batch([X_realA, x_fake_b], y_fake)
            g_loss, _, _ = self.gan.train_on_batch(X_realA, [y_real, X_realB])
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

            if (i + 1) % (bat_per_epo * 10) == 0:
                summarize_performance(i, self.generator, dataset)

    def predict(self, images):
        return self.generator.predict(images)

    def save_generator(self, file):
        self.generator.save(file)

    def load_trained_generator(self, file):
        self.generator = load_model(file)


def define_discriminator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)

    in_target_image = Input(shape=image_shape)

    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)

    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)

    if batchnorm:
        g = BatchNormalization()(g, training=True)

    g = LeakyReLU(alpha=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)

    if dropout:
        g = Dropout(0.5)(g, training=True)

    g = Concatenate()([g, skip_in])

    g = Activation('relu')(g)
    return g


def define_generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)

    e1 = define_encoder_block(layer_in=in_image, n_filters=64, batchnorm=False)
    e2 = define_encoder_block(layer_in=e1, n_filters=128)
    e3 = define_encoder_block(layer_in=e2, n_filters=256)
    e4 = define_encoder_block(layer_in=e3, n_filters=512)
    e5 = define_encoder_block(layer_in=e4, n_filters=512)
    e6 = define_encoder_block(layer_in=e5, n_filters=512)
    e7 = define_encoder_block(layer_in=e6, n_filters=512)

    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    d1 = decoder_block(layer_in=b, skip_in=e7, n_filters=512)
    d2 = decoder_block(layer_in=d1, skip_in=e6, n_filters=512)
    d3 = decoder_block(layer_in=d2, skip_in=e5, n_filters=512)
    d4 = decoder_block(layer_in=d3, skip_in=e4, n_filters=512, dropout=False)
    d5 = decoder_block(layer_in=d4, skip_in=e3, n_filters=256, dropout=False)
    d6 = decoder_block(layer_in=d5, skip_in=e2, n_filters=128, dropout=False)
    d7 = decoder_block(layer_in=d6, skip_in=e1, n_filters=64, dropout=False)

    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False

    in_src = Input(shape=image_shape)

    gen_out = g_model(in_src)

    dis_out = d_model([in_src, gen_out])

    model = Model(in_src, [dis_out, gen_out])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def summarize_performance(step, g_model, dataset, n_samples=3):
    [x_real_a, x_real_b], _ = generate_real_samples(dataset, n_samples, 1)
    x_fake_b, _ = generate_fake_samples(g_model, x_real_a, 1)
    x_real_a = (x_real_a + 1) / 2.0
    x_real_b = (x_real_b + 1) / 2.0
    x_fake_b = (x_fake_b + 1) / 2.0
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(x_real_a[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(x_fake_b[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.axis('off')
        plt.imshow(x_real_b[i])

    filename1 = 'plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()

    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


def generate_real_samples(dataset, n_samples, patch_shape):
    train_a, train_b = dataset
    ix = np.random.randint(0, train_a.shape[0], n_samples)
    x1, x2 = train_a[ix], train_b[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [x1, x2], y


def generate_fake_samples(g_model, samples, patch_shape):
    x = g_model.predict(samples)
    y = np.zeros((len(x), patch_shape, patch_shape, 1))
    return x, y


