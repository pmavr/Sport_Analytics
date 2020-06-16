import numpy as np
import cv2
from auxiliary import HoughLines as hl, aux
import time




input_file = "../clips/cutvideo.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()

while success:

    img = hl.extract(frame, [40, 60, 60], [60, 255, 255])
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=2)
    img = hl.remove_white_dots(img)
    img = cv2.bitwise_not(img, img)
    img = hl.remove_white_dots(img)
    img = cv2.Canny(img, 50, 120)
    img = cv2.dilate(img, kernel, iterations=3)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=4)
    img = hl.remove_white_dots(img)
    img = cv2.dilate(img, kernel, iterations=2)

    img_with_hough_lines = hl.houghLines(img, frame)

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
