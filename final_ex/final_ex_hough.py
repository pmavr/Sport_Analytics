import numpy as np
import cv2
from auxiliary import HoughLines as hl, aux
import time




input_file = "../clips/belgium_japan.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()
frame_count = 0
while success:

    img = hl.image_preprocess(frame)

    _, img_with_hough_lines = hl.houghLines(img, frame)

    cv2.imshow('Match Detection', img_with_hough_lines)

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
