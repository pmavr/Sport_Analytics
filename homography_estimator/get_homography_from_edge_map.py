import sys
import cv2
import numpy as np
import scipy.io as sio
import pyflann

from torchvision.transforms import ToTensor, Normalize, Compose

from homography_estimator.siamese.Siamese import Siamese
from homography_estimator.Camera import Camera
from homography_estimator.helper import infer_features_from_edge_map, camera_to_edge_map
import utils


if __name__ == '__main__':

    data = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}database_camera_feature_100.mat')
    features = data['features']
    cameras = data['cameras']

    # get edge map
    edge_map = cv2.imread(f'{utils.get_project_root()}homography_estimator/images_for_testing/014_AB_fake_D.png')
    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.resize(edge_map, (320, 180))

    # forward edge map through siamese & and get embedding
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    siamese = Siamese()
    siamese, _, _ = utils.load_model(siamese, f'{utils.get_homography_estimator_model_path()}siamese_100.pth')

    edge_map_features = infer_features_from_edge_map(siamese, edge_map, transform)

    # find closest emdedding
    flann = pyflann.FLANN()
    result, _ = flann.nn(features, edge_map_features, 1, algorithm="kdtree", trees=8, checks=64)
    id = result[0]

    edge_map_camera_params = cameras[id]

    binary_court = sio.loadmat(f'{utils.get_world_cup_2014_dataset_path()}worldcup2014.mat')
    retrieved_image = camera_to_edge_map(binary_court, edge_map_camera_params)



    sys.exit()
