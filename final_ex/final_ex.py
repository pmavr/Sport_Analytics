import numpy as np
import cv2
from auxiliary import ColorClusters as cc, aux
from auxiliary import ObjectDetector as od
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


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



# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"
input_file = "../clips/belgium_japan.mp4"

training_frames = 2
yolo = od.ObjectDetector()

vs = cv2.VideoCapture(input_file)

boxes = []
idx = 0
for j in range(training_frames):
    success, frame = vs.read()
    frame = cv2.resize(frame, (1280, 720))

    img = od.image_preprocess(frame)
    output = yolo.predict(img)
    objects = yolo.extract_objects(output)

    for (b, _, _, _) in objects:
        box = yolo.to_image(b)
        if box.shape[0] > box.shape[1]:
            boxes.append(box)

team_predictor = cc.train_clustering(boxes, n_clusters=3)

vs = cv2.VideoCapture(input_file)

success, frame = vs.read()

while success:

    frame = cv2.resize(frame, (1280, 720))

    img = od.image_preprocess(frame)
    output = yolo.predict(img)
    objects = yolo.extract_objects(output)

    objects = yolo.merge_overlapping_boxes(objects)

    objects = yolo.determine_team(objects, team_predictor)

    frame = yolo.draw_bounding_boxes(objects)

    cv2.imshow('Match Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, frame = vs.read()

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
