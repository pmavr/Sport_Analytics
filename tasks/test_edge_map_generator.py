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

    predicted_edge_maps_ = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)
    masked_court_images_ = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)
    edge_maps_ = np.zeros((test_loader.total_dataset_size(), 256, 256, 3), dtype=np.uint8)

    model.eval()
    with torch.no_grad():

        for i in range(test_loader.__len__()):
            masked_court_image, edge_map = test_loader[i]
            masked_court_image = masked_court_image.to(device)

            pred = model.infer(masked_court_image)
            pred = tensor2im(pred)
            masked_court_image = tensor2im(masked_court_image)
            edge_map = tensor2im(edge_map)

            predicted_edge_maps_[i, :, :, :] = pred
            masked_court_images_[i, :, :, :] = masked_court_image
            edge_maps_[i, :, :, :] = edge_map

    return masked_court_images_, edge_maps_, predicted_edge_maps_


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_edge_map_generator_output_images(masked_court_images_, edge_maps_, predicted_edge_maps_):
    for i in range(len(masked_court_images_)):
        img_delimiter = np.zeros((256, 2, 3), dtype=np.uint8)
        img_to_be_saved = np.concatenate(
            (masked_court_images_[i], img_delimiter, edge_maps_[i], img_delimiter, predicted_edge_maps_[i]), axis=1)
        cv2.imwrite(f'{utils.get_project_root()}tasks/results/edge_map_generator/court_real_fake_{i}.jpg', img_to_be_saved)


if __name__ == '__main__':
    import sys
    import cv2

    from modules.GrassMaskEstimator import GrassMaskEstimator

    print('Loading World Cup 2014 dataset')
    data = np.load(f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_test_dataset.npz')
    court_images = data['A']
    edge_maps = data['B']

    estimator = GrassMaskEstimator(
        model_filename=f'{utils.get_generated_models_path()}grass_mask_estimator_200.pth')

    masked_court_images = np.zeros_like(court_images)

    for i in range(len(court_images)):
        masked_court_image = estimator(court_images[i])
        masked_court_images[i, :, :, :] = masked_court_image

    test_dataset = Pix2PixDataset(
        image_a_data=masked_court_images,
        image_b_data=edge_maps,
        batch_size=1,
        num_of_batches=masked_court_images.shape[0],
        is_train=False)

    pix2pix = Pix2Pix(is_train=False)

    pix2pix, _, _, _, _, _ = Pix2Pix.load_model(f'{utils.get_generated_models_path()}edge_map_generator_200.pth', pix2pix)

    masked_court_images, edge_maps, predicted_edge_maps = evaluate_model(
        model=pix2pix,
        test_loader=test_dataset)

    # sio.savemat(
    #     f'{utils.get_world_cup_2014_dataset_path()}edge_map_generator_output_50.mat',
    #     {
    #         'court_images': masked_court_images,
    #         'real_edge_maps': edge_maps,
    #         'fake_edge_maps': predicted_edge_maps},
    #     do_compression=True)

    save_edge_map_generator_output_images(masked_court_images, edge_maps, predicted_edge_maps)

    sys.exit()
