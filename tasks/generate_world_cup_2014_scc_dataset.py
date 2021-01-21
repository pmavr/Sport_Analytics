import sys
import cv2
import numpy as np
import glob

import utils


def load_world_cup_dataset():
    paths = [
        {'key': 'edge_map_generator_train_dataset', 'path': f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_train_dataset/'},
        {'key': 'edge_map_generator_val_dataset', 'path': f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_val_dataset/'},
        {'key': 'edge_map_generator_test_dataset', 'path': f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_test_dataset/'},
        {'key': 'grass_mask_estimator_train_dataset', 'path': f'{utils.get_world_cup_2014_scc_dataset_path()}grass_mask_estimator_train_dataset/'}
    ]
    datasets = {}

    for path in paths:
        image_filelist = glob.glob(f"{path.get('path')}*.jpg")
        a_images, b_images = [], []

        for i in range(1, len(image_filelist) + 1):
            img = cv2.imread(f"{path.get('path')}{i}.jpg")
            img_a = img[:, :img.shape[1]//2, :]
            img_b = img[:, img.shape[1]//2:, :]
            a_images.append(img_a)
            b_images.append(img_b)

        dictionary = {
            'A': np.array(a_images),
            'B': np.array(b_images)}

        datasets[path.get('key')] = dictionary

    return datasets


if __name__ == '__main__':

    print('Loading World Cup 2014 SCCvSD dataset')
    datasets = load_world_cup_dataset()

    print('Exporting dataset to files')
    np.savez_compressed(f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_train_dataset',
                        A=datasets.get('edge_map_generator_train_dataset')['A'],
                        B=datasets.get('edge_map_generator_train_dataset')['B'])

    np.savez_compressed(f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_val_dataset',
                        A=datasets.get('edge_map_generator_val_dataset')['A'],
                        B=datasets.get('edge_map_generator_val_dataset')['B'])

    np.savez_compressed(f'{utils.get_world_cup_2014_scc_dataset_path()}edge_map_generator_test_dataset',
                        A=datasets.get('edge_map_generator_test_dataset')['A'],
                        B=datasets.get('edge_map_generator_test_dataset')['B'])

    np.savez_compressed(f'{utils.get_world_cup_2014_scc_dataset_path()}grass_mask_estimator_train_dataset',
                        A=datasets.get('grass_mask_estimator_train_dataset')['A'],
                        B=datasets.get('grass_mask_estimator_train_dataset')['B'])

    sys.exit()
