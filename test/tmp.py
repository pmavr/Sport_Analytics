import numpy as np
import cv2
from auxiliary import ColorClusters as cc
import time


def show_image(img, msg=''):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def discard_black_pixels(a):
    a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])
    return a[((a[:, 2] > 0) & a[:, 1] > 0) & (a[:, 0] > 0)]


def remove_green(img):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([60, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = discard_black_pixels(res)
    return res


images = [cv2.imread('../tmp/{}.jpg'.format(i)) for i in range(17)]

filtered_images = [remove_green(image) for image in images]

dc = cc.ColorClusters(filtered_images[0], 3)
colors = dc.colorClusters()

print(colors)

