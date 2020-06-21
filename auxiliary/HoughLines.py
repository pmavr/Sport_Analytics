import numpy as np
import cv2
import operator, math
from scipy.spatial import distance
from auxiliary import aux, homography as hmg
import itertools


def is_horizontal(theta, delta=.0349066 * 2):
    hor_angle = 1.46608
    return True if (hor_angle - delta) <= theta <= (
            hor_angle + delta) else False  # or (-1 * delta) <= theta <= delta else False


def is_vertical(theta, delta=0.0349066 * 2):
    ver_angle = 1.8326
    return True if (ver_angle - delta) <= theta <= (
            ver_angle + delta) else False  # or (3 * np.pi / 2) - delta <= theta <= (3 * np.pi / 2) + delta else False


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


def get_line_endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1500 * (-b))
    y1 = int(y0 + 1500 * (a))
    x2 = int(x0 - 1500 * (-b))
    y2 = int(y0 - 1500 * (a))
    return (x1, y1), (x2, y2)


def draw_line(img, rho, theta):
    (x1, y1), (x2, y2) = get_line_endpoints(rho, theta)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return img


def drawhoughLinesOnImage(image, houghLines):
    for (r, th) in houghLines:
        image = draw_line(image, r, th)


def houghLines(image, coloured_image):
    houghLines = cv2.HoughLines(image, 1, np.pi / 180, 95)
    houghLinesImage = np.zeros_like(image)

    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])
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
    houghLines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=70, minLineLength=100, maxLineGap=100)

    houghLinesImage = np.zeros_like(image)

    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])
        for l in houghLines:
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


def refine_lines(_lines, pixel_thresh=10, degree_thresh=5):
    filtered_lines = np.zeros([6, 1, 2])
    n2 = 0
    for n1 in range(0, len(_lines)):
        rho, theta = _lines[n1]
        if n1 == 0:
            filtered_lines[n2] = _lines[n1]
            n2 = n2 + 1
        else:
            if rho < 0:
                rho *= -1
                theta -= np.pi
            closeness_rho = np.isclose(rho, filtered_lines[0:n2, 0, 0], atol=pixel_thresh)
            closeness_theta = np.isclose(theta, filtered_lines[0:n2, 0, 1], atol=degree_thresh * np.pi / 180)
            closeness = np.all([closeness_rho, closeness_theta], axis=0)
            if not any(closeness) and n2 < 4:
                filtered_lines[n2] = _lines[n1]
                n2 = n2 + 1
    filtered_lines = filtered_lines.reshape(filtered_lines.shape[0], 2)
    filtered_lines = filtered_lines[(filtered_lines[:, 0] > 0) & (filtered_lines[:, 1] > 0)]
    return filtered_lines


def find_intersection(l1, l2):
    (Ax1, Ay1), (Ax2, Ay2) = get_line_endpoints(l1[0], l1[1])
    (Bx1, By1), (Bx2, By2) = get_line_endpoints(l2[0], l2[1])

    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return np.array([x, y])


def draw_intersection_points(coloured_image, lines):
    intersection_mask = np.zeros_like(coloured_image)

    line_pairs = list(itertools.combinations(lines, 2))
    intersection_points = []

    for pair in line_pairs:
        intersection_points.append(find_intersection(pair[0], pair[1]))

    for p in intersection_points:
        if p is not None:
            cv2.line(intersection_mask, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (255, 255, 255), 20)

    intersection_mask = cv2.cvtColor(intersection_mask, cv2.COLOR_BGR2GRAY)

    corner_mask = np.zeros_like(coloured_image)

    drawhoughLinesOnImage(corner_mask, lines)
    corner_mask = cv2.cvtColor(corner_mask, cv2.COLOR_BGR2GRAY)
    tmp = np.float32(corner_mask)

    dst = cv2.cornerHarris(tmp, 15, 15, 0.04)

    tmp2 = cv2.cvtColor(corner_mask, cv2.COLOR_GRAY2RGB)
    tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_BGR2RGB)
    tmp2[dst > 0.01 * dst.max()] = [0, 255, 0]

    lower_color = np.array([40, 255, 255])
    upper_color = np.array([60, 255, 255])
    hsv = cv2.cvtColor(tmp2, cv2.COLOR_BGR2HSV)
    corner_mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.bitwise_and(intersection_mask, corner_mask)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=15)
    mask = cv2.erode(mask, np.ones((4, 4), np.uint8), iterations=5)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    indices = np.where(mask == 255)
    mask[indices[0], indices[1], :] = [0, 0, 255]

    return blend_images(mask, coloured_image)

#
# frame = cv2.imread('../clips/frame0.jpg')
#
# img = image_preprocess(frame)
#
# lines, image_with_lines = houghLines(img, frame)
#
# hor_lines = list()
# ver_lines = list()
#
# for line in lines:
#     rho, theta = line
#     if is_horizontal(theta):
#         hor_lines.append(line)
#     elif is_vertical(theta):
#         ver_lines.append(line)
#
# ref_hor_lines = refine_lines(hor_lines, 2)
# ref_ver_lines = refine_lines(ver_lines, 2)
#
# # if ver_lines is not None:
# #     drawhoughLinesOnImage(frame, ref_ver_lines)
# # if hor_lines is not None:
# #     drawhoughLinesOnImage(frame, ref_hor_lines)
#
# lines = np.concatenate((ref_ver_lines, ref_hor_lines))
#
# corner_points_image = draw_intersection_points(frame, lines)
#
# aux.show_image(corner_points_image)
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
