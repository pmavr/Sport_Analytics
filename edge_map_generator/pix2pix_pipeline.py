import numpy as np

from edge_map_generator.Pix2Pix import Pix2Pix
from edge_map_generator.prepare_data import prepare_input
from edge_map_generator.helper import plot_images
import utils

if __name__ == '__main__':
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
