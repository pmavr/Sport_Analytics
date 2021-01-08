import sys
import numpy as np
import scipy.io as sio

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from homography_estimator.torch.SiameseDataset import SiameseDataset
from homography_estimator.torch.Siamese import Siamese
import utils



def test_model(
        model,
        test_loader):

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(0))
        model = model.to(device)
        cudnn.benchmark = True

    # model.eval()
    features = []
    with torch.no_grad():
        for i in range(test_loader.__len__()):
            x = test_loader[i]
            x = x.to(device)

            feat = model.feature_numpy(x)
            features.append(feat)
    features = np.vstack((features))
    return features



if __name__ == '__main__':

    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()

    print('[INFO] Loading training data..')
    data = sio.loadmat(f'{world_cup_2014_dataset_path}train_data_10k.mat')
    pivot_images = data['pivot_images']
    positive_images = data['positive_images']
    cameras = data['cameras']

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    test_dataset = SiameseDataset(
        pivot_images,
        positive_images,
        batch_size=50,
        num_of_batches=-1,
        data_transform=transform,
        is_train=False)

    siamese = Siamese()

    siamese = utils.load_model(siamese, f'{utils.get_homography_estimator_model_path()}siamese.pth')

    features = test_model(siamese, test_dataset)

    sio.savemat(
        f'{utils.get_world_cup_2014_dataset_path()}feature_camera_10k.mat',
        {
            'features': features,
            'cameras': cameras},
        do_compression=True)

    sys.exit()

