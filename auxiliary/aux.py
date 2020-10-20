import cv2
import numpy as np

def show_image(img, msg=''):
    """
    Displays an image. Esc char to close window
    :param img: Image to be displayed
    :param msg: Optional message-title for the window
    :return:
    """
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def remove_white_dots(image, iterations=1):
    # do connected components processing
    for j in range(iterations):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)
        # get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= 100:  # keep
                result[labels == i + 1] = 255

        image = result
        image = cv2.bitwise_not(image, image)

    return result


def detect_court(image):
    """
    Detects the area corresponding to the soccer field (green colour spectrum)
    :param image:
    :return:
    """
    lower_color = np.array([35, 75, 60])
    upper_color = np.array([65, 255, 200])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((20, 20), np.uint8))
    img = cv2.bitwise_and(image, image, mask=mask)
    return img
