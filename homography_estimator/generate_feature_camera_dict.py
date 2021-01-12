import sys
import numpy as np
import cv2
import scipy.io as sio

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from homography_estimator.siamese.Siamese import Siamese
from homography_estimator.helper import camera_to_edge_map, infer_features_from_edge_map

import utils


if __name__ == '__main__':

    binary_court = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}worldcup2014.mat')

    data = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}worldcup_sampled_cameras.mat')
    pivot_camera_params = data['pivot_cameras']
    num_of_camera_params = len(pivot_camera_params)

    im_h, im_w = 180, 320
    edge_maps = np.zeros((num_of_camera_params, im_h, im_w), dtype=np.uint8)

    for i in range(num_of_camera_params):
        edge_map = camera_to_edge_map(binary_court, pivot_camera_params[i], img_h=im_h, img_w=im_w)

        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.resize(edge_map, (320, 180))

        edge_maps[i, :, :] = edge_map

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    siamese = Siamese()
    siamese, _, _ = utils.load_model(siamese, f'{utils.get_homography_estimator_model_path()}siamese_100.pth')

    features = np.zeros((num_of_camera_params, siamese.embedding_size), dtype=np.float32)
    for i in range(num_of_camera_params):
        edge_map_features = infer_features_from_edge_map(siamese, edge_maps[i], transform)
        features[i, :] = edge_map_features

    sio.savemat(
        f'{utils.get_world_cup_2014_dataset_path()}database_camera_feature_100.mat',
        {
            'features': features,
            'cameras': pivot_camera_params},
        do_compression=True)

    sys.exit()

