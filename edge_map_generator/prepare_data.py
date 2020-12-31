import cv2
import numpy as np


def prepare_input(images, img_dims=(256, 256)):
    '''Normalize pixel values from [0,255] to [-1,1] then, resize images'''
    images = normalize_images(images)
    images = np.asarray([cv2.resize(img, img_dims) for img in images])
    return images


def normalize_images(x):
    return (x - 127.5) / 127.5
