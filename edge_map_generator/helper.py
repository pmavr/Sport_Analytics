import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils


def prepare_data(data, img_dims=(256, 256)):
    x, y = data['court_images'], data['edge_maps']

    # Normalize images
    x = (x - 127.5) / 127.5
    y = (y - 127.5) / 127.5

    # Resize images according to model input
    court_images = np.asarray([cv2.resize(img, img_dims) for img in x])
    edge_maps = np.asarray([cv2.resize(img, img_dims) for img in y])

    return [court_images, edge_maps]


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


def train(discriminator, generator, gan, dataset, n_epochs=100, n_batch=1):
    n_patch = discriminator.output_shape[1]
    train_a, train_b = dataset
    bat_per_epo = int(len(train_a) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        x_fake_b, y_fake = generate_fake_samples(generator, X_realA, n_patch)
        d_loss1 = discriminator.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = discriminator.train_on_batch([X_realA, x_fake_b], y_fake)
        g_loss, _, _ = gan.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, generator, dataset)


def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']

    for i in range(len(images)):
        plt.subplot(1, 3, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])
    plt.show()


if __name__ == '__main__':
    print('Kalispera')
