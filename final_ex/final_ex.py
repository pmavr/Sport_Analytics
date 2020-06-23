import numpy as np
import cv2
from auxiliary import ColorClusters as cc, aux
from auxiliary import ObjectDetector as od
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# input_file = "../clips/france_belgium.mp4"
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

# team_predictor = cc.kmeans_train_clustering(boxes, n_clusters=3)
team_predictor = cc.kmeans_train_clustering(boxes, n_clusters=3)

vs = cv2.VideoCapture(input_file)

writer = cv2.VideoWriter('../clips/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15.0, (1280, 720), True)

success, frame = vs.read()

while success:

    frame_resized = cv2.resize(frame, (1280, 720))

    img = od.image_preprocess(frame_resized)
    output = yolo.predict(img)
    objects = yolo.extract_objects(output)

    objects = yolo.merge_overlapping_boxes(objects)

    objects = yolo.kmeans_determine_team(objects, team_predictor)

    frame_with_boxes = yolo.draw_bounding_boxes(frame_resized, objects)

    cv2.imshow('Match Detection', frame_with_boxes)
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
writer.release()
cv2.destroyAllWindows()
