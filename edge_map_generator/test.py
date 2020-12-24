import cv2
import numpy as np
import matplotlib.pyplot as plt

from edge_map_generator.pix2pix import Pix2Pix, prepare_input
import utils


if __name__ == '__main__':
    model = Pix2Pix()
    model.load_trained_generator(f'{utils.get_edge_map_generator_model_path()}g_model.h5')


    src_image = []
    src_image.append(cv2.imread(f'{utils.get_project_root()}/datasets/test_court_images/manch_5.jpg'))
    src_image = np.asarray(src_image)
    src_image = prepare_input(src_image)
    gen_image = model.predict(src_image)

    images = np.vstack((src_image, gen_image))
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated']

    for i in range(len(images)):
        plt.subplot(1, 2, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])
    plt.show()
