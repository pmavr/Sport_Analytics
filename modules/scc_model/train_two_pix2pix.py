import time
from train_options import opt
from data_loader import CreateDataLoader
from modules.scc_model.pix2pix_model import Pix2PixModel

import utils

opt = opt()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = Pix2PixModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))

total_steps = 0


grass_hist = {
        'discriminator_real_loss': [],
        'discriminator_fake_loss': [],
        'generator_gan_loss': [],
        'generator_l1_loss': []}
epoch_train_data = len(dataset)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    epoch_discriminator_real_loss = 0.
    epoch_discriminator_fake_loss = 0.
    epoch_generator_gan_loss = 0.
    epoch_generator_l1_loss = 0.

    for data in dataset:
        # data = {
        #     'A': data[0],
        #     'B': data[1],
        #     'A_paths': 'Path_to_img',
        #     'B_paths': 'Path_to_img'
        # }
        # visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        # errors1, errors2 = model.get_current_errors()
        errors1 = model.get_current_errors()

        epoch_discriminator_real_loss += errors1['D_real'].item()
        epoch_discriminator_fake_loss += errors1['D_fake'].item()
        epoch_generator_gan_loss += errors1['G_GAN'].item()
        epoch_generator_l1_loss += errors1['G_L1'].item()


        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    epoch_discriminator_real_loss /= epoch_train_data
    epoch_discriminator_fake_loss /= epoch_train_data
    epoch_generator_gan_loss /= epoch_train_data
    epoch_generator_l1_loss /= epoch_train_data

    grass_hist['discriminator_real_loss'].append(epoch_discriminator_real_loss)
    grass_hist['discriminator_fake_loss'].append(epoch_discriminator_fake_loss)
    grass_hist['generator_gan_loss'].append(epoch_generator_gan_loss)
    grass_hist['generator_l1_loss'].append(epoch_generator_l1_loss)
    
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        utils.save_to_pickle_file(grass_hist, f'checkpoints_dir/grass_hist_{epoch}.pkl')

    print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1}: Duration: {time.time() - epoch_start_time:.2f} "
          f"| Discr. Real Loss: {grass_hist['discriminator_real_loss'][-1]:.5f} "
          f"| Discr. Fake Loss: {grass_hist['discriminator_fake_loss'][-1]:.3f} "
          f"| GAN Loss: {grass_hist['generator_gan_loss'][-1]:.3f} "
          f"| L1 Loss: {grass_hist['generator_l1_loss'][-1]:.3f}")
    model.update_learning_rate()
    
    
