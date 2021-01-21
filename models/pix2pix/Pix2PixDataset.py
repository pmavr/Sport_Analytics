import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import utils


class Pix2PixDataset(Dataset):

    def __init__(self,
                 image_a_data,
                 image_b_data,
                 batch_size,
                 num_of_batches,
                 is_train=True):

        self.image_a_data = image_a_data
        self.image_b_data = image_b_data
        self.batch_size = batch_size
        self.num_of_batches = num_of_batches
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        return self._get_test_item(index)

    def _get_train_item(self, index):
        image_a = self.image_a_data[index]
        image_b = self.image_b_data[index]

        image_a, image_b = self._get_image_pair_with_random_offset(image_a, image_b)
        image_a, image_b = self._get_image_pair_randomly_flipped_horizontally(image_a, image_b)
        image_a, image_b = self._transform_image_pair(image_a, image_b)

        return image_a, image_b

    def _get_test_item(self, index):
        image_a = self.image_a_data[index]
        image_b = self.image_b_data[index]

        image_a, image_b = self._transform_image_pair(image_a, image_b)

        return image_a, image_b

    @staticmethod
    def _get_image_pair_with_random_offset(img_a, img_b):
        w = img_a.shape[0]
        h = img_a.shape[1]
        w_offset = random.randint(0, max(0, w - 256 - 1))
        h_offset = random.randint(0, max(0, h - 256 - 1))

        img_a_out = img_a[h_offset:h_offset + 256, w_offset:w_offset + 256, :]
        img_b_out = img_b[h_offset:h_offset + 256, w_offset:w_offset + 256, :]

        return img_a_out, img_b_out

    @staticmethod
    def _get_image_pair_randomly_flipped_horizontally(img_a, img_b):
        to_be_flipped = random.random() > .5
        img_a_out, img_b_out = img_a, img_b
        if to_be_flipped:
            img_a_out = cv2.flip(img_a, 1)
            img_b_out = cv2.flip(img_b, 1)
        return img_a_out, img_b_out

    def _transform_image_pair(self, img_a, img_b):
        img_a_out = transforms.ToTensor()(img_a)
        img_b_out = transforms.ToTensor()(img_b)

        img_a_out = transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(img_a_out)
        img_b_out = transforms.Normalize(mean=[.5], std=[.5])(img_b_out)

        return torch.stack([img_a_out]), torch.stack([img_b_out])

    def total_dataset_size(self):
        return self.num_of_batches * self.batch_size

    def __len__(self):
        return self.num_of_batches


if __name__ == '__main__':
    import sys
    import numpy as np
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage

    print('Loading World Cup 2014 dataset')
    data = np.load(f'{utils.get_world_cup_2014_scc_dataset_path()}grass_mask_estimator_train_dataset.npz')
    court_images = data['A']
    grass_masks = data['B']

    train_dataset = Pix2PixDataset(
        image_a_data=court_images,
        image_b_data=grass_masks,
        batch_size=32,
        num_of_batches=128,
        is_train=True)

    for i in range(train_dataset.__len__()):
        x1, x2 = train_dataset[i]
        print(f'{x1.shape} {x2.shape}')
        break

    # test_dataset = SiameseDataset(pivot_images, positive_images, batch_size, num_batch, normalize, is_train=False)

    # for i in range(len(train_dataset)):
    #     x, _ = test_dataset[i]
    #     print(f'{x.shape}')
    #     break

    sys.exit()
