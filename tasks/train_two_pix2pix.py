import time
from train_options import opt
from data_loader import CreateDataLoader
from models.two_pix2pix.models import create_model

import utils

opt = opt()
assert(opt.model == 'two_pix2pix')
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)

grass_hist = {
        'discriminator_real_loss': [],
        'discriminator_fake_loss': [],
        'generator_gan_loss': [],
        'generator_l1_loss': []}

edge_hist = {
        'discriminator_real_loss': [],
        'discriminator_fake_loss': [],
        'generator_gan_loss': [],
        'generator_l1_loss': []}

epoch_train_data = len(dataset)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    grass_epoch_discriminator_real_loss = 0.
    grass_epoch_discriminator_fake_loss = 0.
    grass_epoch_generator_gan_loss = 0.
    grass_epoch_generator_l1_loss = 0.

    edge_epoch_discriminator_real_loss = 0.
    edge_epoch_discriminator_fake_loss = 0.
    edge_epoch_generator_gan_loss = 0.
    edge_epoch_generator_l1_loss = 0.

    for data in dataset:

        model.set_input(data)
        model.optimize_parameters()

        grass_error, edge_error = model.get_current_errors()
        grass_epoch_discriminator_real_loss += grass_error['D_real'].item() / epoch_train_data
        grass_epoch_discriminator_fake_loss += grass_error['D_fake'].item() / epoch_train_data
        grass_epoch_generator_gan_loss += grass_error['G_GAN'].item() / epoch_train_data
        grass_epoch_generator_l1_loss += grass_error['G_L1'].item() / epoch_train_data

        edge_epoch_discriminator_real_loss += edge_error['D_real'].item() / epoch_train_data
        edge_epoch_discriminator_fake_loss += edge_error['D_fake'].item() / epoch_train_data
        edge_epoch_generator_gan_loss += edge_error['G_GAN'].item() / epoch_train_data
        edge_epoch_generator_l1_loss += edge_error['G_L1'].item() / epoch_train_data


    grass_hist['discriminator_real_loss'].append(grass_epoch_discriminator_real_loss)
    grass_hist['discriminator_fake_loss'].append(grass_epoch_discriminator_fake_loss)
    grass_hist['generator_gan_loss'].append(grass_epoch_generator_gan_loss)
    grass_hist['generator_l1_loss'].append(grass_epoch_generator_l1_loss)

    edge_hist['discriminator_real_loss'].append(edge_epoch_discriminator_real_loss)
    edge_hist['discriminator_fake_loss'].append(edge_epoch_discriminator_fake_loss)
    edge_hist['generator_gan_loss'].append(edge_epoch_generator_gan_loss)
    edge_hist['generator_l1_loss'].append(edge_epoch_generator_l1_loss)

    
    if epoch % opt.save_epoch_freq == 0:
        print(f'saving the model at the end of epoch {epoch}')
        model.save('latest')
        model.save(epoch)
        utils.save_to_pickle_file(grass_hist, f'checkpoints_dir/grass_hist_{epoch}.pkl')
        utils.save_to_pickle_file(edge_hist, f'checkpoints_dir/edge_hist_{epoch}.pkl')

    print('======================\nGrass mask model:')
    print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1} "
          f"| Discr. Real Loss: {grass_hist['discriminator_real_loss'][-1]:.5f} "
          f"| Discr. Fake Loss: {grass_hist['discriminator_fake_loss'][-1]:.3f} "
          f"| GAN Loss: {grass_hist['generator_gan_loss'][-1]:.3f} "
          f"| L1 Loss: {grass_hist['generator_l1_loss'][-1]:.3f}")

    print('Edge map model:')
    print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1} "
          f"| Discr. Real Loss: {edge_hist['discriminator_real_loss'][-1]:.5f} "
          f"| Discr. Fake Loss: {edge_hist['discriminator_fake_loss'][-1]:.3f} "
          f"| GAN Loss: {edge_hist['generator_gan_loss'][-1]:.3f} "
          f"| L1 Loss: {edge_hist['generator_l1_loss'][-1]:.3f}")

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
    
    
