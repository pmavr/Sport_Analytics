import cv2
import numpy as np
import sys

from operator import itemgetter
from numpy import ones, vstack
from numpy.linalg import lstsq
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from auxiliary import aux


def draw_lines_on_frame(lines, frame):
    tmp_frame = np.copy(frame)
    m = 1500
    for vx, vy, x0, y0 in lines:
        cv2.line(tmp_frame, (x0 - m * vx[0], y0 - m * vy[0]), (x0 + m * vx[0], y0 + m * vy[0]), (0, 0, 255), 3)
    return tmp_frame


def draw_hough_lines_on_frame(lines, frame):
    tmp_frame = np.copy(frame)
    for p1, p2, x0, y0 in lines:
        cv2.line(tmp_frame, p1, p2, (0, 0, 255), 1)
    return tmp_frame


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

    if hough_lines is not None:
        hough_lines = hough_lines.reshape(hough_lines.shape[0], hough_lines.shape[2])

    return hough_lines


def get_slope_and_intercept(point1, point2):
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    return round(m, 2), round(c, 2)


def get_lines(hough_lines):
    lines = []
    for point1_x, point1_y, point2_x, point2_y in hough_lines:
        line_endpoint1, line_endpoint2 = (point1_x, point1_y), (point2_x, point2_y)
        slope, intercept = get_slope_and_intercept(line_endpoint1, line_endpoint2)
        lines.append([line_endpoint1, line_endpoint2, slope, intercept])

    return lines


def divide_lines_by_slope(lines):

    # group A lines are perpendicular to group B lines
    line_group_A, line_group_B = [], []

    for line in lines:
        slope = line[2]
        line_group_A.append(line) if slope > 0 else line_group_B.append(line)

    return line_group_A, line_group_B


def find_outliers(values, bandwidth=5, outlier_thresh=.01):
    values = np.array(values)
    values = values.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    scores = kde.score_samples(values)
    threshold = np.quantile(scores, outlier_thresh)

    index = np.where(scores <= threshold)
    return list(index[0])


def remove_outliers_by_slope(lines):
    slopes = [slope for (_, _, slope, _) in lines]
    slope_outlier_indexes = find_outliers(slopes, bandwidth=.5, outlier_thresh=.01)
    lines_without_outliers = [line for idx, line in enumerate(lines) if idx not in slope_outlier_indexes]
    return lines_without_outliers


def remove_outliers_by_intercept(lines):
    intercepts = [intercept for (_, _, _, intercept) in lines]
    intercept_outlier_indexes = find_outliers(intercepts, bandwidth=.5, outlier_thresh=.05)
    lines_without_outliers = [line for idx, line in enumerate(lines) if idx not in intercept_outlier_indexes]
    return lines_without_outliers


def remove_outlier_lines(lines):
    lines_without_slope_outliers = remove_outliers_by_slope(lines)

    lines_without_outliers = remove_outliers_by_intercept(lines_without_slope_outliers)

    return lines_without_outliers


def find_group_boundaries(values):
    values = np.array(values)
    values = values.reshape(-1, 1)
    max_value = max(values)
    min_value = min(values)

    kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(values)

    s = np.linspace(min_value, max_value)

    e = kde.score_samples(s.reshape(-1, 1))

    mi = argrelextrema(e, np.less)[0]
    return s[mi]


def batch_lines(lines, intercept_groups):
    intercept_criterion = itemgetter(3)
    lines = sorted(lines, key=intercept_criterion)

    intercept_groups_iter = iter(intercept_groups)
    group = next(intercept_groups_iter, sys.maxsize)
    merged_lines = [[lines[0]]]

    for line in lines[1:]:
        _, _, _, intercept = line
        if intercept <= group:
            merged_lines[-1].append(line)
        else:
            merged_lines.append([line])
            group = next(intercept_groups_iter, sys.maxsize)

    return merged_lines


def merge_lines_within_batches(grouped_duplicates):
    fit_lines = []
    for group in grouped_duplicates:
        tmp1 = [x for x, _, _, _ in group]
        tmp2 = [x for _, x, _, _ in group]
        line_endpoints = np.array(tmp1 + tmp2)
        fit_line = cv2.fitLine(line_endpoints, distType=cv2.DIST_L2, param=0, reps=.01, aeps=.01)
        fit_lines.append(fit_line)
    return fit_lines


def merge_similar_lines(lines):
    intercepts = [intercept for (_, _, _, intercept) in lines]

    intercept_groups = find_group_boundaries(intercepts)

    batches = batch_lines(lines, intercept_groups)

    merged_lines = merge_lines_within_batches(batches)

    return merged_lines


def detect_court_lines(video_frame):

    hough_lines = get_hough_lines(video_frame)

    if hough_lines is None:
        return None

    lines = get_lines(hough_lines)

    lines_a, lines_b = divide_lines_by_slope(lines)

    lines_a = remove_outlier_lines(lines_a)
    lines_b = remove_outlier_lines(lines_b)

    lines_a_refined = merge_similar_lines(lines_a)
    lines_b_refined = merge_similar_lines(lines_b)

    return lines_a_refined + lines_b_refined


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

        # video_source.display_frame(video_frame)
        cv2.imshow('Playing video...', video_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

        video_frame = video_source.get_frame()

    video_source.clean_up()
