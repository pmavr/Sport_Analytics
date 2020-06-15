import numpy as np
import cv2
from auxiliary import ColorClusters as cc
from auxiliary import HoughLines as hl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def show_image(img, msg=''):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def plot_clusters(clrs, labels, n_clusters=2):
    print('[INFO] Plot dominant object colors')
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 125, 125), (125, 0, 125)]
    fig = plt.figure()
    ax = Axes3D(fig)
    for label, pix in zip(labels, clrs):
        ax.scatter(pix[0], pix[1], pix[2], color=cc.rgb_to_hex(colors[label]))
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    plt.show()


def extract_objects(image, layer_outputs):
    h, w = image.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > desired_conf:

                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if x < 0 or y < 0:
                    continue

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return boxes, confidences, classIDs


# input_file = "../clips/cutvideo.mp4"
input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"

labels_file = "../yolo_files/yolov3.txt"
config_file = "../yolo_files/yolov3.cfg"
weights_file = "../yolo_files/yolov3.weights"
desired_conf = .5
desired_thres = .3

CLASS_PERSON = 0
CLASS_BALL = 32

LABELS = open(labels_file).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] Loading YOLO from disk...")
yolo = cv2.dnn.readNetFromDarknet(config_file, weights_file)
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = yolo.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()

while success:

    img_with_hough_lines = hl.houghLines(frame)

    cv2.imshow('Match Detection', img_with_hough_lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, frame = vs.read()

    time.sleep(0.05)

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
