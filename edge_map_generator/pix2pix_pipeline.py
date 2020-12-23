import numpy as np

from data_preparation.load_data import load_world_cup_dataset
from edge_map_generator.pix2pix import Pix2Pix, prepare_input
from edge_map_generator.helper import plot_images
import utils

if __name__ == '__main__':
    model_path = utils.get_edge_map_generator_model_path()
    print('Loading World Cup 2014 dataset')
    train_data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset.npz')

    input_shape = (256, 256, 3)

    model = Pix2Pix(input_shape)

    # print('Training pix2pix model...')
    # model.train(data=train_data, n_epochs=1)

    model.save_generator(f'{model_path}g_model_v2.h5')

    model.load_trained_generator(f'{utils.get_edge_map_generator_model_path()}g_model.h5')

    court_images, edge_maps = train_data['court_images'], train_data['edge_maps']
    src_image, tar_image = court_images[:1], edge_maps[:1]

    gen_image = model.predict(src_image)

    src_image = prepare_input(src_image)
    tar_image = prepare_input(tar_image)

    plot_images(src_img=src_image, tar_img=tar_image, gen_img=gen_image)
