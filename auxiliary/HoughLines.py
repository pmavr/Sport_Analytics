import numpy as np
import cv2
import operator, math
from scipy.spatial import distance
from auxiliary import aux, homography as hmg
import itertools


def is_horizontal(line_angle, degree_tol=4):
    horizontal_angle = 84.
    th = line_angle * 180. / np.pi
    return True if (horizontal_angle - degree_tol) <= th <= (horizontal_angle + degree_tol) else False


def is_vertical(line_angle, degree_tol=4):
    vertical_angle = 110.
    th = line_angle * 180. / np.pi
    return True if (vertical_angle - degree_tol) <= th <= (vertical_angle + degree_tol) else False


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
    line_length = 2200
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + line_length * (-b))
    y1 = int(y0 + line_length * (a))
    x2 = int(x0 - line_length * (-b))
    y2 = int(y0 - line_length * (a))
    return (x1, y1), (x2, y2)


def get_line_midpoints(rho, theta):
    (x1, y1), (x2, y2) = get_line_endpoints(rho, theta)
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)

    x = int((x + x2) / 2)
    y = int((y + y2) / 2)
    return x, y


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


def image_preprocess(image):
    lower_color = np.array([40, 60, 60])
    upper_color = np.array([60, 255, 225])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = aux.remove_white_dots(mask, iterations=2)
    img = cv2.bitwise_and(image, image, mask=mask)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    img = aux.remove_white_dots(img, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.Canny(img, 500, 200)
    # img = remove_white_dots(img, iterations=2)
    return img


def lines_are_close(midA, midB, rtol):
    closeness = np.isclose(midA, midB, rtol=rtol)
    return closeness[0] and closeness[1]


def refine_lines(_lines, rtol=.1):
    _lines = [[l, get_line_midpoints(l[0], l[1]), True] for l in _lines]

    # for (current_line, current_mid, current_keep) in _lines:
    #     cv2.line(frame, current_mid, current_mid, (255, 255, 255), 20)

    for current_idx, (current_line, current_mid, current_keep) in enumerate(_lines):
        if not current_keep:
            continue
        for checked_idx, (checked_line, checked_mid, checked_keep) in enumerate(_lines):
            if not checked_keep or current_idx == checked_idx:
                continue
            if lines_are_close(current_mid, checked_mid, rtol):
                _lines[checked_idx][2] = False

    refine_lines = [line for (line, _, keep) in _lines if keep]

    # for (current_line, current_mid, current_keep) in _lines:
    #     if current_keep:
    #         cv2.line(frame, current_mid, current_mid, (255, 255, 255), 20)
    #         draw_line(frame, current_line[0], current_line[1])

    return refine_lines


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
    line_pairs = list(itertools.combinations(lines, 2))
    intersection_points = []

    for pair in line_pairs:
        intersection_points.append(find_intersection(pair[0], pair[1]))

    for p in intersection_points:
        if p is not None:
            cv2.line(coloured_image, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (255, 255, 0), 10)

    # mask = cv2.cvtColor(intersection_mask, cv2.COLOR_BGR2GRAY)

    # # mask = intersection_mask
    # mask = cv2.bitwise_and(intersection_mask, corner_mask)
    # mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=15)
    # mask = cv2.erode(mask, np.ones((4, 4), np.uint8), iterations=5)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # indices = np.where(mask == 255)
    # mask[indices[0], indices[1], :] = [0, 0, 255]

    # return blend_images(mask, coloured_image)
    return coloured_image

#
# frame = cv2.imread('../clips/frame4.jpg')
#
# img = image_preprocess(frame)
#
# lines, image_with_lines = houghLines(img, frame)
#
# hor_lines = list()
# ver_lines = list()
#
# # aux.show_image(image_with_lines)
#
# for line in lines:
#     rho, theta = line
#     if is_horizontal(theta):
#         hor_lines.append(line)
#     elif is_vertical(theta):
#         ver_lines.append(line)
#
# # if ver_lines is not None:
# #     drawhoughLinesOnImage(frame, ver_lines)
# # if hor_lines is not None:
# #     drawhoughLinesOnImage(frame, hor_lines)
#
#
# ref_ver_lines = refine_lines(ver_lines, rtol=.125)
# ref_hor_lines = refine_lines(hor_lines, rtol=.125)
#
#
# if ref_hor_lines is not None:
#     drawhoughLinesOnImage(frame, ref_hor_lines)
# if ref_ver_lines is not None:
#     drawhoughLinesOnImage(frame, ref_ver_lines)
#
# # aux.show_image(frame)
# lines = []
# for line in ref_hor_lines:
#     lines.append(line)
# for line in ref_ver_lines:
#     lines.append(line)
# #
# #
# image_with_intersections = draw_intersection_points(frame, lines)
#
# # for (x, y) in tmp:
# #     cv2.line(frame, (x, y), (x, y), (255, 255, 255), 20)
#
# # aux.show_image(image_with_intersections)
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
