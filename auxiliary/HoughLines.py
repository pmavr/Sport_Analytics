import numpy as np
import cv2
import matplotlib.pyplot as plt
from auxiliary import aux
from test.asdf import FieldTransform, is_edge_line, merge_similar_lines


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

    minLineLength = 100
    maxLineGap = 5
    houghLines = cv2.HoughLinesP(image, dis_reso, theta, threshold=50, minLineLength=100, maxLineGap=100)
    # houghLines = cv2.HoughLines(image, dis_reso, theta, threshold)
    #
    houghLines.shape = (houghLines.shape[0], houghLines.shape[2])
    #
    # Remove lines on border and merge remaining multiple detections
    good_lines = np.array([l for l in houghLines if not is_edge_line(l, image.shape)])
    houghLines = good_lines[0]
    for i in range(1, len(good_lines)):
        houghLines = merge_similar_lines(good_lines[i], houghLines)

    houghLinesImage = np.zeros_like(image)  # create and empty image

    if houghLines is not None:
        for i in range(0, len(houghLines)):
            l = houghLines[i]
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1, cv2.LINE_AA)

    # houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_RGB2GRAY)

    # houghLines = cv2.HoughLines(image, dis_reso, theta, threshold)

    # houghLines2 = cv2.HoughLinesP(houghLinesImage, dis_reso, theta, 200, minLineLength, maxLineGap)
    # houghLinesImage2 = np.zeros_like(image)
    # if houghLines2 is not None:
    #     for i in range(0, len(houghLines2)):
    #         l = houghLines2[i][0]
    #         cv2.line(houghLinesImage2, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2, cv2.LINE_AA)
    #
    # aux.show_image(houghLinesImage, 'houghLinesImage')
    # aux.show_image(houghLinesImage2, 'houghLinesImage2')

    # aux.show_image(houghLinesImage)
    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_GRAY2RGB)
    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_BGR2RGB)
    orginalImageWithHoughLines = blend_images(houghLinesImage, coloured_image)  # add two images together, using image blending
    # aux.show_image(orginalImageWithHoughLines, 'houghLinesImage2')
    return orginalImageWithHoughLines


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

    # img = cv2.GaussianBlur(image, (5, 5), 0)

    img = extract(image, [40, 60, 60], [60, 255, 255])
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)

    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=2)
    img = remove_white_dots(img, iterations=2)

    img = cv2.Canny(img, 50, 120)

    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=5)
    img = remove_white_dots(img, iterations=2)

    img = cv2.dilate(img, kernel, iterations=1)

    return img

def image_preprocess2(image):
    lower_color = np.array([40, 60, 60])
    upper_color = np.array([60, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = remove_white_dots(mask, iterations=2)
    img = cv2.bitwise_and(image, image, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = remove_white_dots(img, iterations=2)
    return img


# frame = cv2.imread('../clips/frame05.jpg')
# # ft = FieldTransform()
# # output = ft(frame)
#
# img = image_preprocess2(frame)
#
# output = houghLines(img, frame)
#
# aux.show_image(output,'Image with lines')
