import time
from train_options import opt
from data_loader import CreateDataLoader
from modules.scc_model.models import create_model

import numpy as np
from models.pix2pix.Pix2PixDataset import Pix2PixDataset
from visualizer import Visualizer
import utils

opt = opt()
# assert(opt.model == 'two_pix2pix')
# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
# visualizer = Visualizer(opt)
total_steps = 0

data = np.load(f'{utils.get_world_cup_2014_scc_dataset_path()}grass_mask_estimator_train_dataset.npz')
court_images = data['A']
grass_masks = data['B']
dataset = Pix2PixDataset(
    image_a_data=court_images,
    image_b_data=grass_masks,
    batch_size=1,
    num_of_batches=court_images.shape[0],
    is_train=True)
dataset_size = len(dataset)


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

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    

    # for i, data in enumerate(dataset):
    for data in dataset:
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        # visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        
        if total_steps % opt.print_freq == 0:
            # errors1, errors2 = model.get_current_errors()
            errors1 = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            grass_hist['discriminator_real_loss'].append(errors1['D_real'].item())
            grass_hist['discriminator_fake_loss'].append(errors1['D_fake'].item())
            grass_hist['generator_gan_loss'].append(errors1['G_GAN'].item())
            grass_hist['generator_l1_loss'].append(errors1['G_L1'].item())

            # edge_hist['discriminator_real_loss'].append(errors2['D_real'].item())
            # edge_hist['discriminator_fake_loss'].append(errors2['D_fake'].item())
            # edge_hist['generator_gan_loss'].append(errors2['G_GAN'].item())
            # edge_hist['generator_l1_loss'].append(errors2['G_L1'].item())

            print('======================\nGrass mask model:')
            print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1}: Duration: {epoch_iter:.2f} "
                  f"| Discr. Real Loss: {grass_hist['discriminator_real_loss'][-1]:.5f} "
                  f"| Discr. Fake Loss: {grass_hist['discriminator_fake_loss'][-1]:.3f} "
                  f"| GAN Loss: {grass_hist['generator_gan_loss'][-1]:.3f} "
                  f"| L1 Loss: {grass_hist['generator_l1_loss'][-1]:.3f}")

            # print('Edge map model:')
            # print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1}: Duration: {epoch_iter:.2f} "
            #       f"| Discr. Real Loss: {edge_hist['discriminator_real_loss'][-1]:.5f} "
            #       f"| Discr. Fake Loss: {edge_hist['discriminator_fake_loss'][-1]:.3f} "
            #       f"| GAN Loss: {edge_hist['generator_gan_loss'][-1]:.3f} "
            #       f"| L1 Loss: {edge_hist['generator_l1_loss'][-1]:.3f}")

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        iter_data_time = time.time()
        
    
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        utils.save_to_pickle_file(grass_hist, f'checkpoints_dir/grass_hist_{epoch}.pkl')
        utils.save_to_pickle_file(edge_hist, f'checkpoints_dir/edge_hist_{epoch}.pkl')

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
    
    
