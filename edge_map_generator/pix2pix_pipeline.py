
import numpy as np

from data_preparation.load_data import load_world_cup_dataset
from edge_map_generator.pix2pix import define_generator, define_discriminator, define_gan, load_trained_model
from edge_map_generator.helper import train, plot_images, prepare_data
import utils


if __name__ == '__main__':

    model_path = utils.get_edge_map_generator_model_path()
    print('Loading World Cup 2014 dataset')
    train_data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset.npz')

    input_shape = (256, 256)
    court_images, edge_maps = prepare_data(train_data, img_dims=input_shape)

    # d_model = define_discriminator(input_shape)
    # g_model = define_generator(input_shape)
    # gan_model = define_gan(g_model, d_model, input_shape)
    #
    # # print('Training pix2pix model...')
    # train(d_model, g_model, gan_model, (court_images, edge_maps), n_epochs=1)
    # g_model.save(f'{model_path}g_model_v2.h5')

    g_model = load_trained_model()

    src_image, tar_image = court_images[:1], edge_maps[:1]
    gen_image = g_model.predict(src_image)
    plot_images(src_img=src_image, tar_img=tar_image, gen_img=gen_image)