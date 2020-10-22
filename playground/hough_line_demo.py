import cv2
import numpy as np
import math
from auxiliary import HoughLines as hl, aux
from numpy import ones,vstack
from numpy.linalg import lstsq
from sklearn import cluster


def get_coef_intercept(point1, point2):
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return round(m,2), round(c,2)


def get_angle(point1, point2):
    """ Calculates the angle of a corner [0,0] - point1 - point2"""

    delta_x = point1[0] - point2[0]
    delta_y = point1[1] - point2[1]
    theta_degrees = math.atan2(delta_y, delta_x) * 180. / np.pi
    return round(theta_degrees, 2)


def houghLinesP(image):
    houghLines = cv2.HoughLinesP(image,
                                 rho=1,  # Distance resolution of the accumulator in pixels.
                                 theta=np.pi / 180,  # Angle resolution of the accumulator in radians.
                                 threshold=100,     # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
                                 minLineLength=75,  # Minimum line length. Line segments shorter than that are rejected.
                                 maxLineGap=75  # Maximum allowed gap between points on the same line to link them.
                                 )
    tmp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    houghLinesImage = np.zeros_like(tmp_image)

    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])
        for l in houghLines:
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv2.LINE_AA)

    return houghLines, houghLinesImage


def cluster_by_distance(lines, max_distance):

    clusters = [[lines[0]]]
    for line in lines[1:]:
        _, _, _, cur_intercept = line
        _, _, _, prev_intercept = clusters[-1][-1]
        if abs(cur_intercept - prev_intercept) <= max_distance:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return clusters


input_file = "../clips/belgium_japan.mp4"
# input_file = "../clips/aris_aek.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
vs = cv2.VideoCapture(input_file)
success, frame = vs.read()

while success:
    img = cv2.resize(frame, (1280, 720))

    img_c = np.copy(img)

    img_c = aux.detect_court(img_c)

    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    _, img_c = cv2.threshold(img_c, 150, 200, cv2.THRESH_BINARY)
    img_c = cv2.Canny(img_c, 500, 200)

    houghLines = cv2.HoughLinesP(img_c,
                                 rho=1,  # Distance resolution of the accumulator in pixels.
                                 theta=np.pi / 180,  # Angle resolution of the accumulator in radians.
                                 threshold=100,
                                 # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
                                 minLineLength=75,  # Minimum line length. Line segments shorter than that are rejected.
                                 maxLineGap=75  # Maximum allowed gap between points on the same line to link them.
                                 )
    img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)

    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])

        line_list = []
        for l in houghLines:
            x = (l[0], l[1])
            y = (l[2], l[3])
            line_list.append([x, y, get_angle(x, y)])

        line_list_plus = []
        line_list_minus = []
        for l in line_list:
            line_list_plus.append(l) if l[2] > 0 else line_list_minus.append(l)


        plus_coef_intercepts_list = []
        for p1, p2, _ in line_list_plus:
            coef, intercept = get_coef_intercept(p1, p2)
            plus_coef_intercepts_list.append([p1, p2, coef, intercept])

        minus_coef_intercepts_list = []
        for p1, p2, _ in line_list_minus:
            coef, intercept = get_coef_intercept(p1, p2)
            minus_coef_intercepts_list.append([p1, p2, coef, intercept])


        from operator import itemgetter
        plus_coef_intercepts_list = sorted(plus_coef_intercepts_list, key=itemgetter(3))
        minus_coef_intercepts_list = sorted(minus_coef_intercepts_list, key=itemgetter(3))

        grouped_lines_plus = cluster_by_distance(plus_coef_intercepts_list, max_distance=50)
        grouped_lines_minus = cluster_by_distance(minus_coef_intercepts_list, max_distance=50)
        fit_lines = []

        for group in grouped_lines_plus:
            tmp1 = [x for x, _, _, _ in group]
            tmp2 = [x for _, x, _, _ in group]
            line_endpoints = np.array(tmp1 + tmp2)
            fit_line = cv2.fitLine(line_endpoints, distType=cv2.DIST_L2, param=0, reps=.01, aeps=.01)
            fit_lines.append(fit_line)

        for group in grouped_lines_minus:
            tmp1 = [x for x, _, _, _ in group]
            tmp2 = [x for _, x, _, _ in group]
            line_endpoints = np.array(tmp1 + tmp2)
            fit_line = cv2.fitLine(line_endpoints, distType=cv2.DIST_L2, param=0, reps=.01, aeps=.01)
            fit_lines.append(fit_line)

        m = 1500
        for vx, vy, x0, y0 in fit_lines:
            cv2.line(img_c, (x0-m*vx[0], y0-m*vy[0]), (x0+m*vx[0], y0+m*vy[0]), (0, 0, 255), 1)

    # show dual images
    concat = np.concatenate((img, img_c), axis=1)
    concat = cv2.resize(concat, (0, 0), None, .75, 1)
    cv2.imshow('Display lines', concat)


    # writer.write(frame_with_boxes)
    # video play - pause - quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

    success, frame = vs.read()

print("[INFO] cleaning up...")
vs.release()
# writer.release()
cv2.destroyAllWindows()
