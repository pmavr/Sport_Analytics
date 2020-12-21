from numpy import asarray

from data_preparation.load_data import load_world_cup_dataset
from edge_map_generator.pix2pix import define_generator, define_discriminator, define_gan
from edge_map_generator.helper import resize_image, train, load_real_samples
import utils


if __name__ == '__main__':

    model_path = utils.get_edge_map_generator_model_path()

    print('Loading World Cup 2014 dataset')
    train_data, _ = load_world_cup_dataset()
    court_images = train_data.get('court_images')
    edge_maps = train_data.get('edge_maps')

    court_images = asarray([resize_image(img, (256, 256)) for img in court_images])
    edge_maps = asarray([resize_image(img, (256, 256)) for img in edge_maps])

    image_shape = court_images.shape[1:]

    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    print('Training pix2pix model...')
    train(d_model, g_model, gan_model, (court_images, edge_maps), n_epochs=100)
    g_model.save(f'{model_path}g_model.h5')

    # g_model = load_model(f'{model_path}g_model.h5')
    # ix = np.random.randint(0, len(dataset[0]), 1)
    # src_image, tar_image = dataset[0][ix], dataset[1][ix]
    # gen_image = g_model.predict(src_image)
    # plot_images(src_img=src_image, tar_img=tar_image, gen_img=gen_image)