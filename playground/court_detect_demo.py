
import cv2
from cv2 import dnn_superres
import numpy as np
from auxiliary import aux

def detect_court(image):
    lower_color = np.array([35, 100, 60])
    upper_color = np.array([75, 255, 200])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((20, 20), np.uint8))
    img = cv2.bitwise_and(image, image, mask=mask)
    return img


def detect_white_line_pixels(image):
    lower_color = np.array([35, 0, 145])
    upper_color = np.array([75, 140, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel=np.ones((2, 2), np.uint8))
    img = cv2.bitwise_and(image, image, mask=mask)
    return img


# input_file = "../clips/belgium_japan.mp4"
input_file = "../clips/aris_aek.mp4"
# input_file = "../clips/chelsea_pmanchester.mp4"
vs = cv2.VideoCapture(input_file)
success, frame = vs.read()

while success:
    img = cv2.resize(frame, (1280, 720))

    img_c = detect_court(img)

    img_c = detect_white_line_pixels(img_c)

    # show dual images
    concat = np.concatenate((img,img_c), axis=1)
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



