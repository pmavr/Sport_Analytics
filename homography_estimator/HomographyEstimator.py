
import cv2
import pyflann
import scipy.io as sio

from torchvision.transforms import ToTensor, Normalize, Compose

from homography_estimator.Camera import Camera
from models.siamese.Siamese import Siamese
import Helper
import utils


class HomographyEstimator:

    def __init__(self,
                 model_filename,
                 feature_pose_database_filename,
                 binary_court_filename):
        self.binary_court = sio.loadmat(binary_court_filename)

        data = sio.loadmat(feature_pose_database_filename)
        self.features = data['features']
        self.cameras = data['cameras']

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.0188], std=[0.128])])

        siamese = Siamese()
        self.model, _, _ = Siamese.load_model(model_filename, siamese)

    def infer_camera_from_edge_map(self, edge_map_):
        edge_map_features = Helper.infer_features_from_edge_map(self.model, edge_map_, self.transform)

        # find closest emdedding
        flann = pyflann.FLANN()
        result, _ = flann.nn(self.features, edge_map_features, 1, algorithm="kdtree", trees=8, checks=64)

        id = result[0]
        edge_map_camera_params = self.cameras[id]
        return Camera(edge_map_camera_params)


if __name__ == '__main__':
    import sys

    # get edge map
    # original_edge_map = cv2.imread(f'{utils.get_project_root()}datasets/test/court_edge_maps/001_AB_fake_D.png')
    # original_edge_map = cv2.imread(f'{utils.get_project_root()}datasets/test/court_edge_maps/003_AB_fake_D.png')
    # original_edge_map = cv2.imread(f'{utils.get_project_root()}datasets/test/court_edge_maps/013_AB_fake_D.png')
    original_edge_map = cv2.imread(f'{utils.get_project_root()}datasets/test/court_edge_maps/014_AB_fake_D.png')
    original_edge_map = cv2.resize(original_edge_map, (320, 180))
    original_edge_map = cv2.cvtColor(original_edge_map, cv2.COLOR_BGR2GRAY)
    # _, original_edge_map = cv2.threshold(original_edge_map, 10, 255, cv2.THRESH_BINARY)

    estimator = HomographyEstimator(
        model_filename=f'{utils.get_generated_models_path()}siamese_10.pth',
        feature_pose_database_filename=f'{utils.get_world_cup_2014_dataset_path()}database_camera_feature_10.mat',
        binary_court_filename=f'{utils.get_world_cup_2014_dataset_path()}worldcup2014.mat')

    estimated_camera = estimator.infer_camera_from_edge_map(original_edge_map)

    estimated_edge_map = Helper.camera_to_edge_map(
        estimator.binary_court,
        estimated_camera)

    utils.show_image([original_edge_map, estimated_edge_map], ['edge map', 'estimated_edge_map'])

    sys.exit()
