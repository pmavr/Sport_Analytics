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


# def get_horizontal_flip_homography(court_img, homography_orig):
#     '''
#     Get the homography for an image after it 's flipped horizontally
#     :param court_img: the original image for which you want to calculate the homography of its flipped counterpart
#     :param homography_orig: homography of the original image
#     :return:
#     '''
#     # taken from Homayounfar et al. - 2017
#
#     # court dimensions in yards
#     field_width = 114.83
#     field_height = 74.37
#     img_width = court_img.shape[1]
#     img_height = court_img.shape[0]
#
#     homography_rev = np.array([[-1, 0, img_width], [0, 1, 0], [0, 0, 1]])
#     homography_rev_inv = homography_rev
#     homography_rev_model = np.array([[-1, 0, field_width], [0, 1, 0], [0, 0, 1]])
#
#     homography = homography_orig.dot(homography_rev_inv)
#     homography = homography_rev_model.dot(homography)
#     return homography
#
#
# def augment_soccer_dataset(initial_path, relative_path):
#     court_image_path = f'{initial_path}{relative_path}court_images/'
#     homographies_path = f'{initial_path}{relative_path}homography_matrices/'
#     grass_mat_path = f'{initial_path}{relative_path}grass_mats/'
#
#     dataset_length = os.listdir(court_image_path).__len__()
#
#     for i in range(1, dataset_length + 1):
#         court_img = cv2.imread('{}{}.jpg'.format(court_image_path, i))
#
#         mat = loadmat('{}{}_grass_gt.mat'.format(grass_mat_path, i))
#         grass_mask = mat.get('grass')
#
#         with open('{}{}.homographyMatrix'.format(homographies_path, i)) as homography_file:
#             data = homography_file.readlines()
#         matrix = read_homography_matrix(data)
#
#         flipped_court_img = cv2.flip(court_img, 1)
#         flipped_grass_mask = cv2.flip(grass_mask, 1)
#         flipped_mat = {'grass': flipped_grass_mask}
#         flipped_court_homography = get_horizontal_flip_homography(court_img, matrix)
#
#         flipped_image_code = i + dataset_length
#         cv2.imwrite(f'{court_image_path}{flipped_image_code}.jpg', flipped_court_img)
#         savemat(f'{grass_mat_path}{flipped_image_code}_grass_gt.mat', flipped_mat, do_compression=True)
#         np.savetxt(f'{homographies_path}{flipped_image_code}.homographyMatrix',
#                    flipped_court_homography, fmt='%.7e', delimiter='\t')


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
