import sys
import numpy as np
import cv2
import scipy.io as sio

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

import Helper
from homography_estimator.siamese.Siamese import Siamese


import utils


def convert_camera_params_to_edge_maps(binary_court_, camera_params, edge_map_h=180, edge_map_w=320):
    num_of_camera_params = camera_params.shape[0]
    edge_maps_ = np.zeros((num_of_camera_params, edge_map_h, edge_map_w), dtype=np.uint8)

    for i in range(num_of_camera_params):
        edge_map_ = Helper.camera_to_edge_map(binary_court_, camera_params[i], img_h=edge_map_h, img_w=edge_map_w)
        edge_map_ = cv2.resize(edge_map_, (edge_map_w, edge_map_h))
        edge_map_ = cv2.cvtColor(edge_map_, cv2.COLOR_BGR2GRAY)

        edge_maps_[i, :, :] = edge_map_
    return edge_maps_


def extract_features_from_edge_maps(model, data_transform, edge_maps_):
    num_of_camera_params = edge_maps_.shape[0]
    features = np.zeros((num_of_camera_params, model.embedding_size), dtype=np.float32)
    for i in range(num_of_camera_params):
        edge_map_features = Helper.infer_features_from_edge_map(model, edge_maps_[i], data_transform)
        features[i, :] = edge_map_features
    return features


if __name__ == '__main__':


    binary_court = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}worldcup2014.mat')
    data = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}worldcup_sampled_cameras.mat')
    pivot_camera_params = data['pivot_cameras']

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    siamese = Siamese()
    siamese, optimizer, history = Siamese.load_model(f'{utils.get_homography_estimator_model_path()}siamese_500.pth',
                                                   siamese)

    edge_maps = convert_camera_params_to_edge_maps(
        binary_court, pivot_camera_params,
        edge_map_h=siamese.input_shape[1], edge_map_w=siamese.input_shape[2])

    features = extract_features_from_edge_maps(
        model=siamese, data_transform=transform, edge_maps_=edge_maps)

    sio.savemat(
        f'{utils.get_world_cup_2014_dataset_path()}database_camera_feature_500.mat',
        {
            'features': features,
            'cameras': pivot_camera_params},
        do_compression=True)

    sys.exit()

