import cv2
import numpy as np
import math
import sys

from operator import itemgetter
from numpy import ones, vstack
from numpy.linalg import lstsq
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from auxiliary import aux


def detect_court(video_frame):
    """
    Detects the area corresponding to the soccer field (green colour spectrum)
    """
    lower_color = np.array([35, 75, 60])
    upper_color = np.array([65, 255, 200])
    hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((20, 20), np.uint8))
    img = cv2.bitwise_and(video_frame, video_frame, mask=mask)
    return img


def get_frame_edges(video_frame):
    frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 150, 200, cv2.THRESH_BINARY)
    frame = cv2.Canny(frame, 500, 200)
    return frame


def get_hough_lines(video_frame):
    frame = get_frame_edges(video_frame)

    hough_lines = cv2.HoughLinesP(
        frame,
        rho=1,  # Distance resolution of the accumulator in pixels.
        theta=np.pi / 180,  # Angle resolution of the accumulator in radians.
        threshold=100,
        # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
        minLineLength=75,
        # Minimum line length. Line segments shorter than that are rejected.
        maxLineGap=75  # Maximum allowed gap between points on the same line to link them.
        )
    if hough_lines is None:
        return None
    else:
        hough_lines = hough_lines.reshape(hough_lines.shape[0], hough_lines.shape[2])
        return hough_lines


def get_coefficient_and_intercept(point1, point2):
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    return round(m, 2), round(c, 2)


def get_lines(hough_lines):
    lines = []
    for point1_x, point1_y, point2_x, point2_y in hough_lines:
        line_endpoint1, line_endpoint2 = (point1_x, point1_y), (point2_x, point2_y)
        coef, intercept = get_coefficient_and_intercept(line_endpoint1, line_endpoint2)
        lines.append([line_endpoint1, line_endpoint2, coef, intercept])

    return lines


def divide_lines_by_perpendicularity(lines):

    # group A lines are perpendicular to group B lines
    line_group_A, line_group_B = [], []

    for line in lines:
        line_group_A.append(line) if line[2] > 0 else line_group_B.append(line)

    return line_group_A, line_group_B


def get_line_boundaries(values, range):
    _values = values.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(_values)

    s = np.linspace(0, range)

    e = kde.score_samples(s.reshape(-1, 1))

    mi = argrelextrema(e, np.less)[0]
    return s[mi]


def cluster_lines(lines, ranges):
    lines = sorted(lines, key=itemgetter(3))

    ranges_iter = iter(ranges)
    range = next(ranges_iter, sys.maxsize)
    clusters = [[lines[0]]]
    for line in lines[1:]:
        _, _, _, intercept = line
        if intercept <= range:
            clusters[-1].append(line)
        else:
            clusters.append([line])
            range = next(ranges_iter, sys.maxsize)
    return clusters


def group_duplicate_lines(lines):
    lines_intercepts = [i for (_, _, _, i) in lines]
    lines_intercepts = np.array(lines_intercepts)
    max_intercept_value = max(lines_intercepts)

    cluster_criteria = get_line_boundaries(lines_intercepts, max_intercept_value)

    line_groups = cluster_lines(lines, cluster_criteria)

    return line_groups


def remove_duplicate_lines(grouped_duplicates):
    fit_lines = []
    for group in grouped_duplicates:
        tmp1 = [x for x, _, _, _ in group]
        tmp2 = [x for _, x, _, _ in group]
        line_endpoints = np.array(tmp1 + tmp2)
        fit_line = cv2.fitLine(line_endpoints, distType=cv2.DIST_L2, param=0, reps=.01, aeps=.01)
        fit_lines.append(fit_line)
    return fit_lines


def draw_lines_on_frame(lines, frame):
    tmp_frame = np.copy(frame)
    m = 1500
    for vx, vy, x0, y0 in lines:
        cv2.line(tmp_frame, (x0 - m * vx[0], y0 - m * vy[0]), (x0 + m * vx[0], y0 + m * vy[0]), (0, 0, 255), 1)
    return tmp_frame


def draw_hough_lines_on_frame(lines, frame):
    tmp_frame = np.copy(frame)
    for p1, p2, x0, y0 in lines:
        cv2.line(tmp_frame, p1, p2, (0, 0, 255), 1)
    return tmp_frame


def detect_court_lines(video_frame):

    hough_lines = get_hough_lines(video_frame)

    if hough_lines is None:
        return None

    lines = get_lines(hough_lines)

    lines_A, lines_B = divide_lines_by_perpendicularity(lines)

    lines_A_grouped_duplicates = group_duplicate_lines(lines_A)
    lines_B_grouped_duplicates = group_duplicate_lines(lines_B)

    lines_A_refined = remove_duplicate_lines(lines_A_grouped_duplicates)
    lines_B_refined = remove_duplicate_lines(lines_B_grouped_duplicates)

    return lines_A_refined + lines_B_refined


def identify_intersection_points_for_court_lines(court_lines):
    return None


if __name__ == "__main__":

    video_source = aux.VideoSource()
    video_frame = video_source.get_frame()

    while video_frame is not None:

        frame_with_court_filtered = detect_court(video_frame)

        court_lines = detect_court_lines(frame_with_court_filtered)

        if court_lines is not None:
            video_frame = draw_lines_on_frame(court_lines, video_frame)

        # court_lines_intersection_points = identify_intersection_points_for_court_lines(court_lines)

        video_source.display_frame(video_frame)

        video_frame = video_source.get_frame()

    video_source.clean_up()
