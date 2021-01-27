from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
