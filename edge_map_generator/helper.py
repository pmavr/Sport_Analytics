
import numpy as np
import matplotlib.pyplot as plt

import utils


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
