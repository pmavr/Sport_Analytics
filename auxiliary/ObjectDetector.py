import numpy as np
import cv2
from auxiliary import ColorClusters as cc, aux


class ObjectDetector:
    CLASS_PERSON = 0
    CLASS_BALL = 32

    def __init__(self):
        self.labels_file = "../yolo_files/yolov3.txt"
        self.config_file = "../yolo_files/yolov3.cfg"
        self.weights_file = "../yolo_files/yolov3.weights"
        self.desired_conf = .5
        self.desired_thres = .3

        self.LABELS = open(self.labels_file).read().strip().split("\n")

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        print("[INFO] Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.layer_names)

    def merge_overlapping_boxes(self, box_list, conf_list):
        return cv2.dnn.NMSBoxes(box_list, conf_list, self.desired_conf, self.desired_thres)

    def extract_objects(self, image, layer_outputs):
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

                if confidence > self.desired_conf:

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

    def draw_bounding_boxes(self, image, box_list, non_overlapping_boxes_IDs, conf_list, class_list):
        if len(non_overlapping_boxes_IDs) > 0:
            for i in non_overlapping_boxes_IDs.flatten():
                (x, y) = (box_list[i][0], box_list[i][1])
                (w, h) = (box_list[i][2], box_list[i][3])
                color = [int(c) for c in self.COLORS[class_list[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[class_list[i]], conf_list[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


def image_preprocess(image):
    lower_color = np.array([35, 100, 60])
    upper_color = np.array([60, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.dilate(mask, np.ones((6, 6), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((50, 50), np.uint8))

    return cv2.bitwise_and(image, image, mask=mask)


def determine_team(boxes, classes):
    pass


input_file = "../clips/belgium_japan.mp4"
yolo = ObjectDetector()

vs = cv2.VideoCapture(input_file)

boxes = []
idx = 0
for j in range(2):
    success, frame = vs.read()
    frame = cv2.resize(frame, (1280, 720))

    img = image_preprocess(frame)
    output = yolo.predict(img)
    box_list, conf_list, class_list = yolo.extract_objects(frame, output)

    for b in box_list:
        box = frame[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]
        if box.shape[0] > box.shape[1]:
            boxes.append(box)
            # cv2.imwrite('../tmp/{}.jpg'.format(idx), box)
            # idx += 1

a, b = cc.train_clustering(boxes, n_clusters=3)

#
#
#
#
#
# # class_list = determine_team(box_list, class_list)
#
# non_overlapping_boxes_IDs = yolo.merge_overlapping_boxes(box_list, conf_list)
#
# frame = yolo.draw_bounding_boxes(frame, box_list, non_overlapping_boxes_IDs, conf_list, class_list)
#
# aux.show_image(frame)
#
#
#
#
# #
# # imgs = []
# # tmp = []
# # keep = []
# # h = 0
# #
# for idx, b in enumerate(box_list):
#     cv2.imwrite('../tmp/{}.jpg'.format(idx), frame[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])
#
#
#
#
#
#
#
# frame = cv2.imread('../clips/frame2.jpg')
# frame = cv2.resize(frame, (1280, 720))
#
# img = image_preprocess(frame)
# output = yolo.predict(img)
# box_list, conf_list, class_list = yolo.extract_objects(frame, output)
#
# class_list = determine_team(box_list, class_list)
#
# non_overlapping_boxes_IDs = yolo.merge_overlapping_boxes(box_list, conf_list)
#
# frame = yolo.draw_bounding_boxes(frame, box_list, non_overlapping_boxes_IDs, conf_list, class_list)
#
# aux.show_image(frame)
