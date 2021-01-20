
import torch
from torch.utils.data.dataset import Dataset, T_co
from torchvision import transforms

import utils


class Pix2PixDataset(Dataset):

    def __init__(self,
                 court_image_data,
                 grass_mask_data,
                 batch_size,
                 num_of_batches,
                 data_transform,
                 is_train=True):

        self.court_image_data = court_image_data
        self.grass_mask_data = grass_mask_data
        self.batch_size = batch_size
        self.num_of_batches = num_of_batches
        self.data_transform = data_transform
        self.is_train = is_train

    def __getitem__(self, index):

        court_image = self.court_image_data[index]
        grass_mask = self.grass_mask_data[index]

        court_image = self.data_transform(court_image)
        court_image = transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(court_image)

        grass_mask = self.data_transform(grass_mask)
        grass_mask = transforms.Normalize(mean=[.5], std=[.5])(grass_mask)

        return torch.stack([court_image]), torch.stack([grass_mask])

    def total_dataset_size(self):
        return self.num_of_batches * self.batch_size

    def __len__(self):
        return self.num_of_batches


if __name__ == '__main__':
    import sys
    import numpy as np
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage

    model_path = utils.get_grass_mask_estimator_model_path()
    print('Loading World Cup 2014 dataset')
    data = np.load(f'{utils.get_world_cup_2014_dataset_path()}world_cup_2014_train_dataset.npz')
    court_images = data['court_images']
    grass_masks = data['grass_masks']

    transform = Compose([
        ToTensor(),
        Resize((256, 256))])

    train_dataset = Pix2PixDataset(
        court_image_data=court_images,
        grass_mask_data=grass_masks,
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
