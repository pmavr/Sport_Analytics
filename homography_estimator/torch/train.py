import sys
from time import time
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.nn import PairwiseDistance
from torch.optim import Adam
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from homography_estimator.torch.SiameseDataset import SiameseDataset
from homography_estimator.torch.Siamese import Siamese
from homography_estimator.torch.ContrastiveLoss import ContrastiveLoss
import utils
sns.set()


def fit_model(
        model,
        opt_func,
        loss_func,
        train_loader,
        val_loader=None,
        num_of_epochs=10,
        scheduler=None,
        silent=False):

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(0))
        model = model.to(device)
        loss_func = ContrastiveLoss(margin=1.0).cuda(device)
        cudnn.benchmark = True

    l2_distance = PairwiseDistance(p=2)

    hist = {
        'num_of_epochs': num_of_epochs,
        'train_loss': [],
        'val_loss': [],
        'positive_distance': [],
        'negative_distance': [],
        'distance_ratio': []}

    start_time = time()
    epoch_train_data = train_loader.total_dataset_size()

    for epoch in range(num_of_epochs):
        epoch_start_time = time()
        model.train()

        epoch_train_loss = 0.
        positive_distance = 0.
        negative_distance = 0.

        for i in range(train_loader.__len__()):
            x1, x2, y_true = train_loader[i]
            x1, x2, y_true = x1.to(device), x2.to(device), y_true.to(device)

            f1, f2 = siamese(x1, x2)

            opt_func.zero_grad()
            loss = loss_func(f1, f2, y_true)
            loss.backward()
            opt_func.step()

            epoch_train_loss += loss.item()

            distance = l2_distance(f1, f2)
            for j in range(len(y_true)):
                if y_true[j] == 1:
                    positive_distance += distance[j]
                elif y_true[j] == 0:
                    negative_distance += distance[j]
                else:
                    assert 0

        epoch_train_loss /= epoch_train_data
        positive_distance /= epoch_train_data
        negative_distance /= epoch_train_data
        distance_ratio = negative_distance / (positive_distance + 0.000001)

        hist['train_loss'].append(epoch_train_loss)
        hist['positive_distance'].append(float(positive_distance))
        hist['negative_distance'].append(float(negative_distance))
        hist['distance_ratio'].append(float(distance_ratio))

        if val_loader:
            model.eval()
            with torch.no_grad():
                [val_x1, val_x2], val_y_true = next(iter(val_loader))
                val_x1, val_x2, val_y_true = val_x1.to(device), val_x2.device(device), val_y_true.to(device)

                val_y_pred = model((val_x1, val_x2))
                val_loss = loss_func(val_y_true, val_y_pred)

                epoch_val_loss = val_loss.item()
                hist['val_loss'].append(epoch_val_loss)

        if scheduler:
            scheduler.step()

        epoch_duration = time() - epoch_start_time

        train_loader.sample_once()

        if not silent:
            print(f"Epoch {epoch+1}/{num_of_epochs}: Duration: {epoch_duration:.2f} "
                  f"| Train Loss: {hist['train_loss'][-1]:.5f} "
                  f"| Positive Dist.: {hist['positive_distance'][-1]:.3f} "
                  f"| Negative Dist.: {hist['negative_distance'][-1]:.3f} "
                  f"| Dist. Ratio: {hist['distance_ratio'][-1]:.3f}")
        else:
            print('.', end='')

    training_time = time() - start_time
    print('\nTotal training time: {:.2f}s'.format(training_time))

    return model, opt_func, hist


def plot_results(history, info):

    epochs_range = [i for i in range(history['num_of_epochs'])]

    fig, (loss_plot, distances, dist_ratio) = plt.subplots(1, 3, figsize=(10, 4))

    loss_plot.plot(epochs_range, history['train_loss'], color='red', label='train loss')
    # loss_plot.plot(epochs_range, history['val_loss'], color='green', label='val loss')
    loss_plot.set_title('Epochs - Loss / {}'.format(info))
    loss_plot.legend()

    distances.plot(epochs_range, history['positive_distance'], color='red', label='positive dist.')
    distances.plot(epochs_range, history['negative_distance'], color='green', label='negative dist.')
    distances.set_title('Epochs - Pos/Neg distance / {}'.format(info))
    distances.legend()

    dist_ratio.plot(epochs_range, history['distance_ratio'], color='red', label='distance ratio')
    dist_ratio.set_title('Epochs - Distance ratio / {}'.format(info))
    dist_ratio.legend()

    plt.show()


if __name__ == '__main__':

    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()

    print('[INFO] Loading training data..')
    data = sio.loadmat(f'{world_cup_2014_dataset_path}train_data_10k.mat')
    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    train_dataset = SiameseDataset(
        pivot_images,
        positive_images,
        batch_size=64,
        num_of_batches=128,
        data_transform=transform,
        is_train=True)

    siamese = Siamese()

    criterion = ContrastiveLoss(margin=1.0)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, siamese.parameters()),
        lr=.01,
        weight_decay=0.000001)

    network, optimizer, history = fit_model(
        model=siamese,
        opt_func=optimizer,
        loss_func=criterion,
        train_loader=train_dataset,
        val_loader=None,
        num_of_epochs=100)

    plot_results(history, info='')

    utils.save_model(network, optimizer, f'{utils.get_homography_estimator_model_path()}siamese.pth')
    utils.save_to_pickle_file(history, f'{utils.get_homography_estimator_model_path()}history.pkl')

    # sys.exit()
