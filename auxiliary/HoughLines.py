import numpy as np
import cv2
import matplotlib.pyplot as plt
from auxiliary import aux
import itertools


def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta, gamma)


def extract(img, lower_range, upper_range):
    lower_color = np.array(lower_range)
    upper_color = np.array(upper_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def drawhoughLinesOnImage(image, houghLines):
    for line in houghLines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)


def houghLines(image, coloured_image):
    # Detect points that form a line
    dis_reso = 1  # Distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # Angular resolution in radians of the Hough grid

    houghLines = cv2.HoughLinesP(image, dis_reso, theta, threshold=100, minLineLength=100, maxLineGap=100)
    # houghLines = cv2.HoughLines(image, dis_reso, theta, threshold)

    houghLinesImage = np.zeros_like(image)  # create and empty image

    if houghLines is not None:
        for i in range(0, len(houghLines)):
            l = houghLines[i][0]
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1, cv2.LINE_AA)

    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_GRAY2RGB)
    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_BGR2RGB)
    orginalImageWithHoughLines = blend_images(houghLinesImage,
                                              coloured_image)  # add two images together, using image blending

    return houghLines, orginalImageWithHoughLines


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


def image_preprocess(image):
    lower_color = np.array([40, 60, 60])
    upper_color = np.array([60, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = remove_white_dots(mask, iterations=2)
    img = cv2.bitwise_and(image, image, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # img = remove_white_dots(img, iterations=2)
    return img


def find_intersection(l1, l2):
    l1 = l1.reshape(-1, 1)
    l2 = l2.reshape(-1, 1)

    # Calculate intercept and gradient of first line
    m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
    b1 = l1[1] - m1 * l1[0]

    # If line is vertical, manually derive intersection
    if l2[0] == l2[2]:
        return np.array([l2[0], m1 * l2[0] + b1])

    # Find intercept and gradient of second line
    m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
    b2 = l2[1] - m2 * l2[0]

    # Calculate intercepts of lines
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    if x < 0 or y < 0 or x > 720 or y > 1280:
        return np.array([0, 0])

    return np.array([x, y])


def get_points(hough_lines):
    intersection_points = list()

    for line_i in hough_lines:
        for line_j in hough_lines:
            intersection_points.append(find_intersection(line_i, line_j))
    return intersection_points


frame = cv2.imread('../clips/frame05.jpg')

img = image_preprocess(frame)

lines, image_with_lines = houghLines(img, frame)

line_pairs = list(itertools.permutations(lines, 2))
intersection_points = list()

for pair in line_pairs:
    intersection_points.append(find_intersection(pair[0], pair[1]))

for dot in intersection_points:
    cv2.line(image_with_lines, (dot[0], dot[1]), (dot[0], dot[1]), (255, 0, 255), 10)

aux.show_image(image_with_lines, 'Image with lines')
