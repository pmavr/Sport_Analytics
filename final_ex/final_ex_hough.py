import numpy as np
import cv2
from auxiliary import HoughLines as hl, aux
# import test.asdf as asdf
import time

input_file = "../clips/belgium_japan.mp4"
# input_file = "../clips/france_belgium.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()

frame_count = 0
while success:
    frame = cv2.resize(frame, (1280, 720))
    img = hl.image_preprocess(frame)

    lines, img_with_hough_lines = hl.houghLines(img, frame)

    hor_lines = list()
    ver_lines = list()

    if lines is not None:
        for line in lines:
            rho, theta = line
            if hl.is_horizontal(theta):
                hor_lines.append(line)
            elif hl.is_vertical(theta):
                ver_lines.append(line)

    ref_hor_lines = hl.refine_lines(hor_lines, rtol=.125)
    ref_ver_lines = hl.refine_lines(ver_lines, rtol=.125)

    # if ref_hor_lines is not None:
    #     hl.drawhoughLinesOnImage(frame, ref_hor_lines)
    # if ref_ver_lines is not None:
    #     hl.drawhoughLinesOnImage(frame, ref_ver_lines)

    lines = []
    for line in ref_hor_lines:
        lines.append(line)
    for line in ref_ver_lines:
        lines.append(line)

    intersection_points = hl.draw_intersection_points(frame, lines)

    cv2.imshow('Match Detection', intersection_points)

    # video play - pause - quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

    success, frame = vs.read()

    # time.sleep(0.01)
    frame_count += 1

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
