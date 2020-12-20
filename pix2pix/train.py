from model import define_discriminator, define_generator, define_gan
from helper import load_real_samples, train, plot_images, summarize_performance
from utils import get_project_root

if __name__ == '__main__':
    initial_path = f'{get_project_root()}/dataset/'
    model_path = f'{initial_path}pix2pix/'
    dataset = load_real_samples('test_maps_256.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    image_shape = dataset[0].shape[1:]

    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)
    train(d_model, g_model, gan_model, dataset, n_epochs=100)
    g_model.save(f'{model_path}g_model.h5')

    # g_model = load_model(f'{model_path}g_model.h5')
    # ix = np.random.randint(0, len(dataset[0]), 1)
    # src_image, tar_image = dataset[0][ix], dataset[1][ix]
    # gen_image = g_model.predict(src_image)
    # plot_images(src_img=src_image, tar_img=tar_image, gen_img=gen_image)

