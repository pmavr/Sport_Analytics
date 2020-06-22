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


input_file = "../clips/belgium_japan.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"


yolo = od.ObjectDetector()

vs = cv2.VideoCapture(input_file)
#
# for j in range(2):
#     success, frame = vs.read()
#
#     output = yolo.predict(frame)
#
#     box_list, _, _ = od.extract_objects(frame, output)
#
#     for b in box_list:
#         if not (h in tmp):
#             imgs.append(frame[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])
#             keep.append(h)
#         h += 1
#
# for i, idxs in zip(imgs, keep):
#     cv2.imwrite('../tmp/{}.jpg'.format(idxs), i)
#
# # show_images(imgs)
#
# team_predictor, team_colors = cc.train_clustering(imgs, n_clusters=2)
#
# # plot_clusters(team_colors, team_predictor.labels_, n_clusters=2)
#
# imgs=[]
# success, frame = vs.read()
# blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
# yolo.setInput(blob)
# output = yolo.forward(layer_names)
#
# box_list, _, _ = extract_objects(frame, output)
#
# for b in box_list:
#     imgs.append(frame[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])
#
# for idxs, i in enumerate(imgs):
#     cv2.imwrite('../tmp/{}.jpg'.format(idxs), i)
#
# start = time.time()
# something = cc.predict_team(imgs, team_predictor)
# end = time.time() - start
# print('Predicted label: {}'.format(something))
# print('Inference in {}s'.format(end))

success, frame = vs.read()

while success:

    frame = cv2.resize(frame, (1280, 720))

    img = od.image_preprocess(frame)

    output = yolo.predict(img)

    box_list, conf_list, class_list = yolo.extract_objects(frame, output)

    non_overlapping_boxes_IDs = yolo.merge_overlapping_boxes(box_list, conf_list)

    frame = yolo.draw_bounding_boxes(frame, box_list, non_overlapping_boxes_IDs, conf_list, class_list)


    cv2.imshow('Match Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, frame = vs.read()

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
