import numpy as np

import torch
from torchvision.transforms import ToTensor, Resize, Compose
import torch.backends.cudnn as cudnn

from grass_mask_estimator.Pix2Pix import Pix2Pix
from grass_mask_estimator.Pix2PixDataset import Pix2PixDataset
import utils


def evaluate_model(model, test_loader):
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        cudnn.benchmark = True

    predicted_masks = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)
    court_images_ = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)
    grass_masks_ = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)

    model.eval()
    with torch.no_grad():

        for i in range(test_loader.__len__()):
            court_image, grass_mask = test_loader[i]
            court_image = court_image.to(device)

            pred = model.infer(court_image)
            pred = tensor2im(pred)
            court_image = tensor2im(court_image)
            grass_mask = tensor2im(grass_mask)

            predicted_masks[i, :, :, :] = pred
            court_images_[i, :, :, :] = court_image
            grass_masks_[i, :, :, :] = grass_mask

    return court_images_, grass_masks_, predicted_masks


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def load_model(filename, model, gen_optimizer=None, discr_optimizer=None, history=None):
    """Load trained model along with its optimizer and training, plottable history."""
    model_components = torch.load(filename)
    model.load_state_dict(model_components['model'])
    if gen_optimizer:
        gen_optimizer.load_state_dict(model_components['generator_opt_func'])
    if discr_optimizer:
        discr_optimizer.load_state_dict(model_components['discriminator_opt_func'])
    if history:
        history = model_components['history']
    return model, gen_optimizer, discr_optimizer, history


if __name__ == '__main__':
    import sys
    import scipy.io as sio

    model_path = utils.get_grass_mask_estimator_model_path()
    print('Loading World Cup 2014 dataset')
    data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_test_dataset.npz')
    court_images = data['court_images']
    grass_masks = data['grass_masks']*255

    transform = Compose([
        ToTensor(),
        Resize((256, 256))])

    test_dataset = Pix2PixDataset(
        court_image_data=court_images,
        grass_mask_data=grass_masks,
        batch_size=1,
        num_of_batches=court_images.shape[0],
        data_transform=transform)

    pix2pix = Pix2Pix(is_train=False)

    pix2pix, _, _, _ = load_model(f'{utils.get_grass_mask_estimator_model_path()}pix2pix_50.pth', pix2pix)

    court_images, grass_masks, predicted_grass_masks = evaluate_model(
        model=pix2pix,
        test_loader=test_dataset)

    sio.savemat(
        f'{utils.get_world_cup_2014_dataset_path()}grass_mask_estimator_output.mat',
        {
            'court_images': court_images,
            'real_grass_masks': grass_masks,
            'fake_grass_masks': predicted_grass_masks},
        do_compression=True)

    sys.exit()
