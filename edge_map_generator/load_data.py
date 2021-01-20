import sys
import cv2
import numpy as np
import scipy.io as sio
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
    mat = sio.loadmat(file)
    return mat.get(element)


def generate_edge_map_from_(image, homography, bin_court):
    w = image.shape[0]
    h = image.shape[1]
    inv_matrix = np.linalg.inv(homography)
    return cv2.warpPerspective(src=bin_court, M=inv_matrix, dsize=(h, w))


def get_horizontal_flip_homography(court_img, homography_orig):
    '''
    Get the homography for an image after it 's flipped horizontally
    :param court_img: the original image for which you want to calculate the homography of its flipped counterpart
    :param homography_orig: homography of the original image
    :return:
    '''
    # taken from Homayounfar et al. - 2017

    # court dimensions in yards
    field_width = 114.83
    field_height = 74.37
    img_width = court_img.shape[1]
    img_height = court_img.shape[0]

    homography_rev = np.array([[-1, 0, img_width], [0, 1, 0], [0, 0, 1]])
    homography_rev_inv = homography_rev
    homography_rev_model = np.array([[-1, 0, field_width], [0, 1, 0], [0, 0, 1]])

    homography = homography_orig.dot(homography_rev_inv)
    homography = homography_rev_model.dot(homography)
    return homography


def augment_dataset(image_dataset, flip=True):
    court_images = image_dataset['court_images']
    edge_maps = image_dataset['edge_maps']
    homographies = image_dataset['homographies']
    grass_masks = image_dataset['grass_masks']

    num_of_images = court_images.shape[0]
    num_of_augm_images = num_of_images * 2
    img_h, img_w = court_images.shape[1], court_images.shape[2]

    augm_court_images = np.zeros((num_of_augm_images, img_h, img_w, 3), dtype=np.uint8)
    augm_edge_maps = np.zeros((num_of_augm_images, img_h, img_w, 3), dtype=np.uint8)
    augm_homographies = np.zeros((num_of_augm_images, 3, 3), dtype=np.float64)
    augm_grass_masks = np.zeros((num_of_augm_images, img_h, img_w), dtype=np.uint8)

    for i in range(num_of_images):
        flipped_court_image = cv2.flip(court_images[i], 1)
        flipped_edge_map = cv2.flip(edge_maps[i], 1)
        flipped_grass_mask = cv2.flip(grass_masks[i], 1)
        flipped_homography = get_horizontal_flip_homography(court_images[i], homographies[i])

        augm_court_images[i, :, :, :] = court_images[i]
        augm_edge_maps[i, :, :, :] = edge_maps[i]
        augm_homographies[i, :, :] = homographies[i]
        augm_grass_masks[i, :, :] = grass_masks[i]

        augm_court_images[num_of_images + i, :, :, :] = flipped_court_image
        augm_edge_maps[num_of_images + i, :, :, :] = flipped_edge_map
        augm_homographies[num_of_images + i, :, :] = flipped_homography
        augm_grass_masks[num_of_images + i, :, :] = flipped_grass_mask

    dataset = {
        'binary_court': image_dataset['binary_court'],
        'court_images':augm_court_images,
        'edge_maps': augm_edge_maps,
        'homographies': augm_homographies,
        'grass_masks': augm_grass_masks}

    return dataset




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
            grass_mask *= 255
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
            'grass_masks': np.array(grass_masks)}

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

    train_dataset = augment_dataset(train_dataset, flip=True)

    print('Exporting dataset to npz files')
    sio.savemat(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset',
                {
                     'court_images': train_dataset['court_images'],
                     'edge_maps': train_dataset['edge_maps'],
                     'homographies': train_dataset['homographies'],
                     'grass_masks': train_dataset['grass_masks']},
                do_compression=True)

    sio.savemat(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_test_dataset',
                {
                     'court_images': test_dataset['court_images'],
                     'edge_maps': test_dataset['edge_maps'],
                     'homographies': test_dataset['homographies'],
                     'grass_masks': test_dataset['grass_masks']},
                do_compression=True)

    sys.exit()
