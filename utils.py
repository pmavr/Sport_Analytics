import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set()


def get_project_root():
    '''
    :return:  path without slash in the end.
    '''
    return f'{Path(__file__).parent}/'


def get_world_cup_2014_dataset_path():
    return f'{get_project_root()}datasets/world_cup_2014/'


def get_world_cup_2014_scc_dataset_path():
    return f'{get_project_root()}datasets/world_cup_2014_scc/'


def get_generated_models_path():
    return f'{get_project_root()}models/generated_models/'


def save_model(model_components, history, filename):
    """Save trained model along with its optimizer and training, plottable history."""
    state = {}
    for key in model_components.keys():
        component = model_components[key]
        state[key] = component.state_dict()
    state['history'] = history
    torch.save(state, filename)


def load_pickle_file(filename):
    data = []
    with open(filename, 'rb') as file:
        while True:
            try:
                data.append(pickle.load(file))
            except EOFError:
                break
        file.close()
        return data


def save_to_pickle_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()


def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]

    for i in range(len(img_list)):
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv2.destroyWindow(msg)


def video_player(video_file):
    """Reproduces a video file. For debugging purposes"""
    import time
    input_file = video_file
    vs = cv2.VideoCapture(input_file)
    success, frame = vs.read()
    while success:
        cv2.imshow('Match Detection', frame)
        time.sleep(0.1)
        # writer.write(frame_with_boxes)
        # video play - pause - quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
        success, frame = vs.read()
    vs.release()
    cv2.destroyAllWindows()


def plot_siamese_results(history, info):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig, (loss_plot, distances, dist_ratio) = plt.subplots(1, 3, figsize=(10, 4))

    loss_plot.plot(epochs_range, history['train_loss'], color='red', label='train loss')
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


def plot_pix2pix_results(history, info):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig, (discriminator_loss, generator_loss) = plt.subplots(1, 2, figsize=(10, 4))

    discriminator_loss.plot(epochs_range, history['discriminator_real_loss'], color='green', label='discr. real loss')
    discriminator_loss.plot(epochs_range, history['discriminator_fake_loss'], color='red', label='discr. fake loss')
    discriminator_loss.set_title('Discriminator Real/Fake Loss')
    discriminator_loss.legend()

    generator_loss.plot(epochs_range, history['generator_gan_loss'], color='red', label='GAN loss')
    generator_loss.plot(epochs_range, history['generator_l1_loss'], color='green', label='L1 loss')
    generator_loss.set_title('GAN/L1 Loss')
    generator_loss.legend()

    plt.title(info)
    plt.show()


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)