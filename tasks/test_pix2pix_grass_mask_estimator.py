import numpy as np

import torch
from torchvision.transforms import ToTensor, Resize, Compose
import torch.backends.cudnn as cudnn

from models.pix2pix.Pix2Pix import Pix2Pix
from models.pix2pix.Pix2PixDataset import Pix2PixDataset
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


def save_grass_mask_estimator_output_images(court_images_, grass_masks_, predicted_grass_masks_):
    for i in range(len(court_images_)):
        img_delimiter = np.zeros((256, 2, 3), dtype=np.uint8)
        img_to_be_saved = np.concatenate(
            (court_images_[i], img_delimiter, grass_masks_[i], img_delimiter, predicted_grass_masks_[i]), axis=1)
        cv2.imwrite(f'{utils.get_project_root()}tasks/results/grass_mask_estimator/court_real_fake_{i}.jpg', img_to_be_saved)


if __name__ == '__main__':
    import sys
    import cv2
    import scipy.io as sio

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

    pix2pix, _, _, _ = Pix2Pix.load_model(f'{utils.get_generated_models_path()}pix2pix_50.pth', pix2pix)

    court_images, grass_masks, predicted_grass_masks = evaluate_model(
        model=pix2pix,
        test_loader=test_dataset)

    sio.savemat(
        f'{utils.get_world_cup_2014_dataset_path()}grass_mask_estimator_output_50.mat',
        {
            'court_images': court_images,
            'real_grass_masks': grass_masks,
            'fake_grass_masks': predicted_grass_masks},
        do_compression=True)

    save_grass_mask_estimator_output_images(court_images, grass_masks, predicted_grass_masks)

    sys.exit()
