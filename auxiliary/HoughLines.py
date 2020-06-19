import numpy as np
import cv2
import matplotlib.pyplot as plt
from auxiliary import aux, homography as hmg
import itertools


def merge_similar_lines(l, lines):
    # Put unique lines in array if theres only 1, for looping
    if len(lines.shape) == 1:
        lines = np.array([lines])
    # Translate line to align with each unique line
    d = np.column_stack(
        ((lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2)
    )
    d = np.tile(d, (1, 2))
    tl = l - d
    # Rotate line to align with each unique line
    xd = lines[:, 2] - lines[:, 0]
    yd = lines[:, 3] - lines[:, 1]
    td = np.sqrt(xd * xd + yd * yd)
    cos_theta = xd / td
    sin_theta = yd / td
    tl = np.column_stack(
        (
            tl[:, 0] * cos_theta + tl[:, 1] * sin_theta,
            tl[:, 1] * cos_theta - tl[:, 0] * sin_theta,
            tl[:, 2] * cos_theta + tl[:, 3] * sin_theta,
            tl[:, 3] * cos_theta - tl[:, 2] * sin_theta,
        )
    )
    # Bounds for the lines to be considered similar
    xb = (
            np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2) / 2
            + 10
    )
    yb = 15
    # Check if line is similar to any unique line
    similar = np.logical_and(abs(tl[:, 1]) < yb, abs(tl[:, 3]) < yb)
    if sum(similar) > 1:
        # If multiple similar lines, take the most similar
        diffs = np.maximum(abs(tl[:, 1]), abs(tl[:, 3]))
        similar[:] = False
        similar[np.argmin(diffs)] = True
    if any(similar):
        # If line is similar, check if it extends beyond current unique line
        # Update unique line to new length if it does
        xb = xb[similar]
        if tl[similar, 0] < -xb or tl[similar, 0] > xb:
            lines[similar, 0:2] = l[0:2]
        if tl[similar, 2] < -xb or tl[similar, 2] > xb:
            lines[similar, 2:4] = l[2:4]
    else:
        # If line is sufficiently differet than other unique lines, add to unique set
        lines = np.concatenate((lines, l.reshape(1, 4)), axis=0)
    return lines


def is_horizontal(theta, delta=.0349066*2):
    hor_angle = 1.46608
    return True if (hor_angle - delta) <= theta <= (hor_angle + delta) else False # or (-1 * delta) <= theta <= delta else False


def is_vertical(theta, delta=0.0349066*2):
    ver_angle = 1.8326
    return True if (ver_angle - delta) <= theta <= (ver_angle + delta) else False # or (3 * np.pi / 2) - delta <= theta <= (3 * np.pi / 2) + delta else False



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


def draw_line(img, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return img


def drawhoughLinesOnImage(image, houghLines):
    for line in houghLines:
        for r, th in line:
            image = draw_line(image, r, th)


def houghLines(image, coloured_image):
    houghLines = cv2.HoughLines(image, 1, np.pi / 180, 95)
    houghLinesImage = np.zeros_like(image)

    if houghLines is not None:
        drawhoughLinesOnImage(houghLinesImage, houghLines)
    # tmp = np.float32(houghLinesImage)
    # dst = cv2.cornerHarris(tmp, 10, 15, 0.04)

    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_GRAY2RGB)
    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_BGR2RGB)
    orginalImageWithHoughLines = blend_images(houghLinesImage, coloured_image)
    # orginalImageWithHoughLines[dst > 0.01 * dst.max()] = [0, 0, 255]

    # aux.show_image(orginalImageWithHoughLines)

    return houghLines, orginalImageWithHoughLines


def houghLinesP(image, coloured_image):
    houghLines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=100)

    houghLinesImage = np.zeros_like(image)

    if houghLines is not None:
        for i in range(0, len(houghLines)):
            l = houghLines[i][0]
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 2, cv2.LINE_AA)

    # tmp = np.float32(houghLinesImage)
    # dst = cv2.cornerHarris(tmp, 10, 15, 0.04)

    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_GRAY2RGB)
    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_BGR2RGB)
    orginalImageWithHoughLines = blend_images(houghLinesImage, coloured_image)
    # orginalImageWithHoughLines[dst>0.01*dst.max()]=[0,0,255]
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
    upper_color = np.array([60, 255, 225])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = remove_white_dots(mask, iterations=2)
    img = cv2.bitwise_and(image, image, mask=mask)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    img = remove_white_dots(img, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.Canny(img, 500, 200)
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
#
#
# frame = cv2.imread('../clips/frame0.jpg')
#
# img = image_preprocess(frame)
#
# lines, image_with_lines = houghLines(img, frame)
#
#
#
# hor_lines = list()
# ver_lines = list()
#
# for idx, line in enumerate(lines):
#     rho, theta = line[0]
#     if asdf.is_horizontal(theta):
#         hor_lines.append([idx, line])
#     elif asdf.is_vertical(theta):
#         ver_lines.append([idx, line])
#
# if ver_lines is not None:
#     drawhoughLinesOnImage(frame, [i[1] for i in ver_lines])
# if hor_lines is not None:
#     drawhoughLinesOnImage(frame, [i[1] for i in hor_lines])
# tmp = np.float32(houghLinesImage)
# dst = cv2.cornerHarris(tmp, 10, 15, 0.04)

# houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_GRAY2RGB)
# houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_BGR2RGB)
# orginalImageWithHoughLines = blend_images(houghLinesImage, coloured_image)

#
# line_pairs = list(itertools.permutations(lines, 2))
# intersection_points = list()

# for pair in line_pairs:
#     intersection_points.append(find_intersection(pair[0], pair[1]))
#
# for dot in intersection_points:
#     cv2.line(image_with_lines, (dot[0], dot[1]), (dot[0], dot[1]), (255, 0, 255), 10)


# court = cv2.imread('../clips/court.png')
#
# lower_color = np.array([20, 60, 60])
# upper_color = np.array([40, 255, 255])
# hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower_color, upper_color)
#
# # hmg.create_homography()
#
#
# aux.show_image(image_with_lines, 'Image with lines')
