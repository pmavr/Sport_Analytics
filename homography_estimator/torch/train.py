import sys
from time import time
import scipy.io as sio
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

import torch
from torch.nn import PairwiseDistance
from torch.optim import Adam
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose

from homography_estimator.torch.SiameseDataset import SiameseDataset
from homography_estimator.torch.Siamese import Siamese
from homography_estimator.torch.ContrastiveLoss import ContrastiveLoss
import utils


def fit_model(
        model,
        opt_func,
        loss_func,
        train_loader,
        val_loader=None,
        num_epochs=10,
        device=torch.device('cpu'),
        scheduler=None,
        silent=False):

    hist = {'train_loss': [], 'val_loss': []}

    start_time = time()

    for epoch in range(num_epochs):
        epoch_start_time = time()
        model.train()
        train_loader.sample_once()

        epoch_train_loss = 0.
        positive_distance = 0.
        negative_distance = 0.

        for i in range(train_loader.__len__()):
            [x1, x2], y_true = train_loader[i]
            x1, x2, y_true = x1.to(device), x2.to(device), y_true.to(device)

            y_pred = siamese((x1, x2))

            opt_func.zero_grad()
            loss = loss_func(y_true, y_pred)
            loss.backward()
            opt_func.step()

            epoch_train_loss += loss.item() / train_loader.__len__()

        hist['train_loss'].append(epoch_train_loss)

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

        if not silent:
            print('Epoch {}/{}: Duration: {:.2f} | Train Loss: {:.4f} | Val. Loss: :.4f'.format(
                epoch+1,
                num_epochs,
                epoch_duration,
                epoch_train_loss#,
                # epoch_val_loss
            ))
        else:
            print('.', end='')

    training_time = time() - start_time
    print('\nTotal training time: {:.2f}s'.format(training_time))

    return model, hist


# def test_model(model, test_data, dev):
#     val_loader = DataLoader(test_data, batch_size = len(test_data))
#     model.eval()
#     with torch.no_grad():
#         val_images, val_labels = next(iter(val_loader))
#         val_images = val_images.reshape(-1, 28*28).to(dev)
#         val_labels = val_labels.to(dev)
#
#         logps = model(val_images)
#         ps = torch.exp(logps)
#         pred_prob, pred_label  = ps.topk(1, dim=1)
#         true_label = val_labels.view(*pred_label.shape)
#         equals = true_label == pred_label
#
#         accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
#
#     print('Test on {} samples - Accuracy: {:.4f}'.format(len(test_data), accuracy))


def plot_results(epochs, history, info):

    epochs_range = [i for i in range(epochs)]

    fig, (loss_plot, acc_plot) = plt.subplots(1, 2, figsize =(12,4))

    loss_plot.plot(epochs_range, history['train_loss'], color='red', label='train loss')
    loss_plot.plot(epochs_range, history['val_loss'], color='green', label='val loss')
    loss_plot.set_title('Epochs - Loss / {}'.format(info))
    loss_plot.legend()

    # acc_plot.plot(epochs_range, history['train_acc'], color='red', label='train acc')
    # acc_plot.plot(epochs_range, history['val_acc'], color='green', label='val acc')
    # acc_plot.set_title('Epochs - Accuracy / {}'.format(info))
    # acc_plot.legend()

    plt.show()


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

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
        batch_size=32,
        num_of_batches=64,
        data_transform=transform,
        is_train=True)

    siamese = Siamese(input_shape=(1, 180, 320)).to(device)

    criterion = ContrastiveLoss(margin=1.0)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, siamese.parameters()),
        lr=.01,
        weight_decay=0.000001)

    network, history = fit_model(
        model=siamese,
        opt_func=optimizer,
        loss_func=criterion,
        train_loader=train_dataset,
        val_loader=None,
        num_epochs=10,
        device=device)

    sys.exit()
