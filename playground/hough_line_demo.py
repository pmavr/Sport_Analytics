import cv2
import numpy as np
import math
from auxiliary import HoughLines as hl, aux


def get_angle(point1, point2):
    """ Calculates the angle of a corner [0,0] - point1 - point2"""
    cv2.norm([0,0], point1)

    delta_x = point2[0] - point1[0]
    delta_y = point1[1] - point2[1]
    theta_radians = math.atan2(delta_y, delta_x)
    return theta_radians * 180. / np.pi


def houghLinesP(image):
    houghLines = cv2.HoughLinesP(image,
                                 rho=1,  # Distance resolution of the accumulator in pixels.
                                 theta=np.pi / 180,  # Angle resolution of the accumulator in radians.
                                 threshold=100,
                                 # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
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

    lines, final_img = houghLinesP(img_c)

    line_list = []
    for l in lines:
        x = (l[0], l[1])
        y = (l[2], l[3])
        line_list.append([l, get_angle(x, y)])

    # show dual images
    concat = np.concatenate((img, final_img), axis=1)
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
