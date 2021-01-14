

import torch
from torch.utils.data.dataset import Dataset, T_co

import utils

class Pix2PixDataset(Dataset):

    def __init__(self,
                 court_image_data,
                 edge_map_data,
                 batch_size,
                 num_of_batches,
                 data_transform,
                 is_train=True):

        self.court_image_data = court_image_data
        self.edge_map_data = edge_map_data
        self.batch_size = batch_size
        self.num_of_batches = num_of_batches
        self.data_transform = data_transform
        self.is_train = is_train

    def _get_train_item(self, index):

        assert index < self.num_of_batches

    def _get_test_item(self, index):
        pass

    def total_dataset_size(self):
        return self.num_of_batches * self.batch_size

    def __getitem__(self, index) -> T_co:
        if self.is_train:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def __len__(self):
        return self.num_of_batches
