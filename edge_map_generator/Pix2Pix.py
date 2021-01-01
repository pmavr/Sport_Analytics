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

import utils

class Pix2Pix:
    # TODO: run model to verify functionality
    # TODO: transfer data generating methods to separate class

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.discriminator = self._define_discriminator()
        self.generator = self._define_generator()
        self.gan = self._define_gan()

    def train(self, x, y, n_epochs=100, n_batch=1):
        dataset = x, y
        n_patch = self.discriminator.output_shape[1]
        train_a, train_b = dataset
        bat_per_epo = int(len(train_a) / n_batch)
        n_steps = bat_per_epo * n_epochs

        for i in range(n_steps):
            [X_realA, X_realB], y_real = self._generate_real_samples(dataset, n_batch, n_patch)
            x_fake_b, y_fake = self._generate_fake_samples(X_realA, n_patch)
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


    def _define_discriminator(self):
        init = RandomNormal(stddev=0.02)

        in_src_image = Input(shape=self.input_shape)

        in_target_image = Input(shape=self.input_shape)

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


    def _define_encoder_block(self, layer_in, n_filters, batchnorm=True):
        init = RandomNormal(stddev=0.02)

        g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)

        if batchnorm:
            g = BatchNormalization()(g, training=True)

        g = LeakyReLU(alpha=0.2)(g)
        return g


    def _decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g, training=True)

        if dropout:
            g = Dropout(0.5)(g, training=True)

        g = Concatenate()([g, skip_in])

        g = Activation('relu')(g)
        return g


    def _define_generator(self):
        init = RandomNormal(stddev=0.02)

        in_image = Input(shape=self.input_shape)

        e1 = self._define_encoder_block(layer_in=in_image, n_filters=64, batchnorm=False)
        e2 = self._define_encoder_block(layer_in=e1, n_filters=128)
        e3 = self._define_encoder_block(layer_in=e2, n_filters=256)
        e4 = self._define_encoder_block(layer_in=e3, n_filters=512)
        e5 = self._define_encoder_block(layer_in=e4, n_filters=512)
        e6 = self._define_encoder_block(layer_in=e5, n_filters=512)
        e7 = self._define_encoder_block(layer_in=e6, n_filters=512)

        b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)

        d1 = self._decoder_block(layer_in=b, skip_in=e7, n_filters=512)
        d2 = self._decoder_block(layer_in=d1, skip_in=e6, n_filters=512)
        d3 = self._decoder_block(layer_in=d2, skip_in=e5, n_filters=512)
        d4 = self._decoder_block(layer_in=d3, skip_in=e4, n_filters=512, dropout=False)
        d5 = self._decoder_block(layer_in=d4, skip_in=e3, n_filters=256, dropout=False)
        d6 = self._decoder_block(layer_in=d5, skip_in=e2, n_filters=128, dropout=False)
        d7 = self._decoder_block(layer_in=d6, skip_in=e1, n_filters=64, dropout=False)

        g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)

        model = Model(in_image, out_image)
        return model


    def _define_gan(self):
        self.discriminator.trainable = False

        in_src = Input(shape=self.input_shape)

        gen_out = self.generator(in_src)

        dis_out = self.discriminator([in_src, gen_out])

        model = Model(in_src, [dis_out, gen_out])

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
        return model

    def _generate_real_samples(self, dataset, n_samples, patch_shape):
        train_a, train_b = dataset
        ix = np.random.randint(0, train_a.shape[0], n_samples)
        x1, x2 = train_a[ix], train_b[ix]
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [x1, x2], y

    def _generate_fake_samples(self, samples, patch_shape):
        x = self.generator.predict(samples)
        y = np.zeros((len(x), patch_shape, patch_shape, 1))
        return x, y

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


if __name__ == '__main__':
    import sys
    from Pix2PixDataset import prepare_input
    from helper import plot_images

    model_path = utils.get_edge_map_generator_model_path()
    print('Loading World Cup 2014 dataset')
    train_data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset.npz')

    court_images = prepare_input(train_data['court_images'])
    edge_maps = prepare_input(train_data['edge_maps'])

    model = Pix2Pix()

    # print('Training pix2pix model...')
    # model.train(x=court_images, y=edge_maps, n_epochs=1)

    # model.save_generator(f'{model_path}g_model_v2.h5')

    model.load_trained_generator(f'{utils.get_edge_map_generator_model_path()}g_model.h5')

    src_image, tar_image = court_images[:1], edge_maps[:1]

    gen_image = model.predict(src_image)

    plot_images(src_img=src_image, tar_img=tar_image, gen_img=gen_image)

    sys.exit()
