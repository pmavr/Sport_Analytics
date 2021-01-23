import sys
from time import time
import numpy as np

import torch
from torchvision.transforms import ToTensor, Resize, Compose
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR

from models.pix2pix.Pix2Pix import Pix2Pix
from models.pix2pix.Pix2PixDataset import Pix2PixDataset
from models.pix2pix.GANLoss import GANLoss
from models.pix2pix.LearningPolicy import LRPolicy
import utils


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def fit_model(model,
              generator_opt_func, discriminator_opt_func,
              gan_loss_func, l1_loss_func,
              generator_sched_func, discriminator_sched_func,
              train_loader, num_of_epochs, num_of_epochs_until_save, silent=False, history=None):

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)

        gan_loss_func.to_cuda()
        l1_loss_func = l1_loss_func.to(device)

        optimizer_to(generator_opt_func, device)
        optimizer_to(discriminator_opt_func, device)

        cudnn.benchmark = True

    if history:
        hist = history
        num_of_trained_epochs = len(hist[next(iter(hist))])
    else:
        hist = {
            'discriminator_real_loss': [],
            'discriminator_fake_loss': [],
            'generator_gan_loss': [],
            'generator_l1_loss': []}
        num_of_trained_epochs = 0
    num_of_epochs += num_of_trained_epochs

    start_time = time()
    epoch_train_data = train_loader.total_dataset_size()

    model.train()
    for epoch in range(num_of_trained_epochs, num_of_epochs):
        epoch_start_time = time()

        epoch_discriminator_real_loss = 0.
        epoch_discriminator_fake_loss = 0.
        epoch_generator_gan_loss = 0.
        epoch_generator_l1_loss = 0.

        for i in range(train_loader.__len__()):
            court_image, grass_mask = train_loader[i]
            court_image = court_image.to(device)
            grass_mask = grass_mask.to(device)

            real_grass_mask, fake_grass_mask = model(court_image, grass_mask)

            discriminator_opt_func.zero_grad()
            discriminator_real_loss, discriminator_fake_loss = model.backward_discriminator(
                real_input=court_image, real_output=real_grass_mask, fake_output=fake_grass_mask,
                gan_criterion=gan_loss_func)
            discriminator_opt_func.step()

            generator_opt_func.zero_grad()
            generator_gan_loss, generator_l1_loss = model.backward_generator(
                real_input=court_image, real_output=real_grass_mask, fake_output=fake_grass_mask,
                gan_criterion=gan_loss_func, l1_criterion=l1_loss_func)
            generator_opt_func.step()

            epoch_discriminator_real_loss += discriminator_real_loss.item()
            epoch_discriminator_fake_loss += discriminator_fake_loss.item()
            epoch_generator_gan_loss += generator_gan_loss.item()
            epoch_generator_l1_loss += generator_l1_loss.item()

        epoch_discriminator_real_loss /= epoch_train_data
        epoch_discriminator_fake_loss /= epoch_train_data
        epoch_generator_gan_loss /= epoch_train_data
        epoch_generator_l1_loss /= epoch_train_data

        hist['discriminator_real_loss'].append(epoch_discriminator_real_loss)
        hist['discriminator_fake_loss'].append(epoch_discriminator_fake_loss)
        hist['generator_gan_loss'].append(epoch_generator_gan_loss)
        hist['generator_l1_loss'].append(epoch_generator_l1_loss)

        epoch_duration = time() - epoch_start_time

        if (epoch + 1) % num_of_epochs_until_save == 0:
            model_components = {
                'model': model,
                'generator_opt_func': generator_opt_func,
                'discriminator_opt_func': discriminator_opt_func,
                'generator_sched_func': generator_sched_func,
                'discriminator_sched_func': discriminator_sched_func}
            utils.save_model(model_components, hist,
                             f"{utils.get_generated_models_path()}edge_map_generator_{len(hist[next(iter(hist))])}.pth")

        if not silent:
            print(f"Epoch {epoch + 1}/{num_of_epochs}: Duration: {epoch_duration:.2f} "
                  f"| Discr. Real Loss: {hist['discriminator_real_loss'][-1]:.5f} "
                  f"| Discr. Fake Loss: {hist['discriminator_fake_loss'][-1]:.3f} "
                  f"| Gener. GAN Loss: {hist['generator_gan_loss'][-1]:.3f} "
                  f"| Gener. L1 Loss: {hist['generator_l1_loss'][-1]:.3f}")
        else:
            print('.', end='')

        discriminator_sched_func.step()
        generator_sched_func.step()

    training_time = time() - start_time
    print('\nTotal training time: {:.2f}s'.format(training_time))

    return model, generator_opt_func, discriminator_opt_func,\
           generator_sched_func, discriminator_sched_func, hist


if __name__ == '__main__':
    model_path = utils.get_generated_models_path()
    print('Loading World Cup 2014 dataset')
    data = np.load(f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_train_dataset.npz')
    masked_court_images = data['A']
    edge_maps = data['B']

    train_dataset = Pix2PixDataset(
        image_a_data=masked_court_images,
        image_b_data=edge_maps,
        batch_size=1,
        num_of_batches=masked_court_images.shape[0],
        is_train=True)

    pix2pix = Pix2Pix(is_train=True)

    criterionGAN = GANLoss(use_lsgan=False, tensor=torch.Tensor)
    criterionL1 = torch.nn.L1Loss()

    generator_optimizer = torch.optim.Adam(
        pix2pix.generator.parameters(), lr=.0002, betas=(.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        pix2pix.discriminator.parameters(), lr=.0002, betas=(.5, 0.999))

    generator_scheduler = LambdaLR(generator_optimizer, LRPolicy(1, 100, 100))
    discriminator_scheduler = LambdaLR(discriminator_optimizer, LRPolicy(1, 100, 100))

    # pix2pix, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, history = Pix2Pix.load_model(
    #     f'{utils.get_generated_models_path()}edge_map_generator_100.pth',
    #     pix2pix, generator_optimizer, discriminator_optimizer,
    #     generator_scheduler, discriminator_scheduler, history=True)

    network, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, history = fit_model(
        model=pix2pix,
        generator_opt_func=generator_optimizer,
        discriminator_opt_func=discriminator_optimizer,
        gan_loss_func=criterionGAN,
        l1_loss_func=criterionL1,
        train_loader=train_dataset,
        generator_sched_func=generator_scheduler,
        discriminator_sched_func=discriminator_scheduler,
        num_of_epochs=200,
        num_of_epochs_until_save=50)
        # history=history)

    model_components = {
        'model': network,
        'generator_opt_func': generator_optimizer,
        'discriminator_opt_func': discriminator_optimizer,
        'generator_sched_func': generator_scheduler,
        'discriminator_sched_func': discriminator_scheduler}
    filename = f"{utils.get_generated_models_path()}edge_map_generator_{len(history[next(iter(history))])}.pth"
    utils.save_model(model_components, history,
                     filename)
    print(f'Saved model at {filename}')

    sys.exit()
