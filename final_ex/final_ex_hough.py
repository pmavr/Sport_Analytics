import numpy as np
import cv2
from auxiliary import HoughLines as hl
import time


def show_image(img, msg=''):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


input_file = "../clips/cutvideo.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()

while success:

    img_with_hough_lines = hl.houghLines(frame)

    cv2.imshow('Match Detection', img_with_hough_lines)

    # video play - pause - quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

    success, frame = vs.read()

    time.sleep(0.01)

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
