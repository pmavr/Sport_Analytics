
import torch
from torch.utils.data.dataset import Dataset, T_co
from torchvision import transforms

import utils


class Pix2PixDataset(Dataset):

    def __init__(self,
                 image_a_data,
                 image_b_data,
                 batch_size,
                 num_of_batches,
                 data_transform,
                 is_train=True):

        self.image_a_data = image_a_data
        self.image_b_data = image_b_data
        self.batch_size = batch_size
        self.num_of_batches = num_of_batches
        self.data_transform = data_transform
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        return self._get_test_item(index)

    def _get_train_item(self, index):
        pass

    def _get_test_item(self, index):
        image_a = self.image_a_data[index]
        image_b = self.image_b_data[index]

        image_a = self.data_transform(image_a)
        image_a = transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(image_a)

        image_b = self.data_transform(image_b)
        image_b = transforms.Normalize(mean=[.5], std=[.5])(image_b)

        return torch.stack([image_a]), torch.stack([image_b])

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

    transform = Compose([
        ToTensor(),
        Resize((256, 256))])

    train_dataset = Pix2PixDataset(
        image_a_data=court_images,
        image_b_data=grass_masks,
        batch_size=32,
        num_of_batches=128,
        data_transform=transform,
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
