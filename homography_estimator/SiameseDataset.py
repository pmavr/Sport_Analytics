import sys
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image

import utils


class SiameseDataset(Sequence):

    def __init__(self,
                 pivot_data,
                 positive_data,
                 batch_size,
                 num_batch,
                 data_transform,
                 is_train=True):
        self.pivot_data = pivot_data
        self.positive_data = positive_data
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.data_transform = data_transform
        self.num_camera = pivot_data.shape[0]

        self.positive_index = []
        self.negative_index = []
        self.is_train = is_train

        if self.is_train:
            self._sample_once()
        else:
            # in testing, loop over all pivot cameras
            self.num_batch = self.num_camera // batch_size
            if self.num_camera % batch_size != 0:
                self.num_batch += 1

    def _sample_once(self):
        self.positive_index = []
        self.negative_index = []
        num = self.batch_size * self.num_batch
        c_set = set([i for i in range(self.num_camera)])
        for i in range(num):
            idx1, idx2 = random.sample(c_set, 2)  # select two indices in random
            self.positive_index.append(idx1)
            self.negative_index.append(idx2)

    def _get_train_item(self, index):
        """
        :param index:
        :return:
        """
        assert index < self.num_batch

        n, c, h, w = self.pivot_data.shape
        batch_size = self.batch_size

        start_index = batch_size * index
        end_index = start_index + batch_size
        positive_index = self.positive_index[start_index:end_index]
        negative_index = self.negative_index[start_index:end_index]

        x1, x2, label = [], [], []

        for i in range(batch_size):
            idx1, idx2 = positive_index[i], negative_index[i]
            pivot = self.pivot_data[idx1].squeeze()
            pos = self.positive_data[idx1].squeeze()
            neg = self.pivot_data[idx2].squeeze()

            pivot = tf.convert_to_tensor(pivot, dtype=tf.float32)
            pos = tf.convert_to_tensor(pos, dtype=tf.float32)
            neg = tf.convert_to_tensor(neg, dtype=tf.float32)

            x1.append(self.data_transform(pivot))
            x1.append(self.data_transform(pivot))
            x2.append(self.data_transform(pos))
            x2.append(self.data_transform(neg))

            label.append(1)
            label.append(0)

        return tf.stack(x1), tf.stack(x2), tf.stack(label)

    def _get_test_item(self, index):
        """
        In testing, the label is hole-fill value, not used in practice.
        :param index:
        :return:
        """
        assert index < self.num_batch

        n, c, h, w = self.pivot_data.shape
        batch_size = self.batch_size

        start_index = batch_size * index
        end_index = min(start_index + batch_size, self.num_camera)
        bsize = end_index - start_index

        x, label_dummy = [], []

        for i in range(start_index, end_index):
            pivot = self.pivot_data[i].squeeze()
            pivot = tf.convert_to_tensor(pivot, dtype=tf.float32)

            x.append(self.data_transform(pivot))
            label_dummy.append(0)

        return tf.stack(x), tf.stack(label_dummy)

    def __len__(self):
        return self.num_batch

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)


if __name__ == '__main__':

    import scipy.io as sio

    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()
    data = sio.loadmat(f'{world_cup_2014_dataset_path}train_data_10k.mat')

    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    normalize = utils.Normalize(mean=[0.0188],
                                std=[0.128])

    batch_size = 32
    num_batch = 64
    train_dataset = SiameseDataset(pivot_images, positive_images, batch_size, num_batch, normalize, is_train=True)

    for i in range(len(train_dataset)):
        x1, x2, label1 = train_dataset[i]
        print(f'{x1.shape} {x2.shape} {label1.shape}')
        break

    test_dataset = SiameseDataset(pivot_images, positive_images, batch_size, num_batch, normalize, is_train=False)

    for i in range(len(train_dataset)):
        x, _ = test_dataset[i]
        print(f'{x.shape}')
        break

    print('train, test dataset size {} {}'.format(len(train_dataset), len(test_dataset)))
    sys.exit()
