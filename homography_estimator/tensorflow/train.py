import sys
import scipy.io as sio
from tensorflow_addons.optimizers import AdamW

from homography_estimator.tensorflow.SiameseDataset import SiameseDataset
from homography_estimator.tensorflow.Siamese import Siamese
from homography_estimator.tensorflow.ContrastiveLoss import ContrastiveLoss
import utils

if __name__ == '__main__':
    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()

    print('[INFO] Loading training data..')
    data = sio.loadmat(f'{world_cup_2014_dataset_path}train_data_10k.mat')
    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    normalize = utils.Normalize(mean=[0.0188], std=[0.128])

    train_generator = SiameseDataset(pivot_images, positive_images,
                          batch_size=32,
                          num_of_batches=64,
                          data_transform=normalize,
                          is_train=True)

    siamese = Siamese(input_shape=(180, 320, 1))

    loss_func = ContrastiveLoss(margin=1.)
    opt_func = AdamW(
        weight_decay=.000001,
        learning_rate=.01,
        epsilon=1e-8
    )

    siamese.compile(
        loss=loss_func,
        optimizer=opt_func
    )

    siamese.fit(
        train_generator,
        # batch_size=32,
        epochs=10)

    sys.exit()
