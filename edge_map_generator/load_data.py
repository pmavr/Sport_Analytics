import sys
import cv2
import numpy as np
from scipy.io import loadmat
from numpy import savez_compressed
import glob
import tarfile

import utils


def load_homography_matrix_from_(file):
    '''Read 3x3 homography matrix from txt file'''
    with open(file) as homography_file:
        data = homography_file.readlines()
    H = np.zeros((3, 3))
    for i in range(len(data)):
        H[i] = np.array([float(x) for x in data[i].strip().split()])
    return H


def load_element_from_mat_(file, element):
    '''Read an element contained inside a .mat file'''
    mat = loadmat(file)
    return mat.get(element)


def generate_edge_map_from_(image, homography, bin_court):
    w = image.shape[0]
    h = image.shape[1]
    inv_matrix = np.linalg.inv(homography)
    return cv2.warpPerspective(src=bin_court, M=inv_matrix, dsize=(h, w))


def load_world_cup_dataset():
    paths = [
        {'key': 'train', 'path': f'{utils.get_world_cup_2014_dataset_path()}raw/train_val/'},
        {'key': 'test', 'path': f'{utils.get_world_cup_2014_dataset_path()}raw/test/'}
    ]
    datasets = {}

    for path in paths:
        binary_court = cv2.imread(f"{utils.get_world_cup_2014_dataset_path()}binary_court.jpg")
        court_image_filelist = glob.glob(f"{path.get('path')}*.jpg")
        court_images, homography_matrices, grass_masks, edge_maps = [], [], [], []

        for i in range(1, len(court_image_filelist) + 1):
            court_image = cv2.imread(f"{path.get('path')}{i}.jpg")
            homography = load_homography_matrix_from_(f"{path.get('path')}{i}.homographyMatrix")
            grass_mask = load_element_from_mat_(f"{path.get('path')}{i}_grass_gt.mat", 'grass')
            edge_map = generate_edge_map_from_(court_image, homography, binary_court)

            court_images.append(court_image)
            homography_matrices.append(homography)
            grass_masks.append(grass_mask)
            edge_maps.append(edge_map)

        dictionary = {
            'binary_court': binary_court,
            'court_images': np.array(court_images),
            'edge_maps': np.array(edge_maps),
            'homographies': np.array(homography_matrices),
            'grass_masks': np.array(grass_masks)
        }
        datasets[path.get('key')] = dictionary

    return datasets.get('train'),  datasets.get('test')


def extract_from_tar(tar_file, extraction_path):
    tar_file = tarfile.open(tar_file)
    tar_file.extractall(path=extraction_path)
    tar_file.close()


if __name__ == '__main__':
    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()

    extract_from_tar(tar_file=f'{world_cup_2014_dataset_path}soccer_data.tar.gz',
                     extraction_path=world_cup_2014_dataset_path)

    print('Loading World Cup 2014 dataset')
    train_dataset, test_dataset = load_world_cup_dataset()

    print('Exporting dataset to npz files')
    savez_compressed(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset',
                     binary_court=train_dataset.get('binary_court'),
                     court_images=train_dataset.get('court_images'),
                     edge_maps=train_dataset.get('edge_maps'),
                     homographies=train_dataset.get('homographies'),
                     grass_masks=train_dataset.get('grass_masks'))

    savez_compressed(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_test_dataset',
                     binary_court=test_dataset.get('binary_court'),
                     court_images=test_dataset.get('court_images'),
                     edge_maps=test_dataset.get('edge_maps'),
                     homographies=test_dataset.get('homographies'),
                     grass_masks=test_dataset.get('grass_masks'))

    # augment_soccer_dataset(initial_path, relative_train_path)
    # augment_soccer_dataset(initial_path, relative_test_path)
    sys.exit()
