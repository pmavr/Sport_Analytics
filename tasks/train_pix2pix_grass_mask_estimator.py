import sys
from time import time
import numpy as np

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from grass_mask_estimator.Pix2Pix import Pix2Pix
from grass_mask_estimator.Pix2PixDataset import Pix2PixDataset
import utils


if __name__ == '__main__':

    model_path = utils.get_grass_mask_estimator_model_path()
    print('Loading World Cup 2014 dataset')
    train_data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset.npz')

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    train_dataset = Pix2PixDataset(
        court_image_data=train_data['court_images'],
        edge_map_data=train_data['edge_maps'],
        batch_size=32,
        num_of_batches=128,
        data_transform=transform,
        is_train=True)

    pix2pix = Pix2Pix(is_train=True)



    sys.exit()
