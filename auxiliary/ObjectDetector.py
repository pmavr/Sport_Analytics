import numpy as np
import cv2
from auxiliary import ColorClusters as cc


class ObjectDetector:
    CLASS_PERSON = 0
    CLASS_BALL = 32
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    def __init__(self):
        self.labels_file = "../yolo_files/yolov3.txt"
        self.config_file = "../yolo_files/yolov3.cfg"
        self.weights_file = "../yolo_files/yolov3.weights"
        self.desired_conf = .5
        self.desired_thres = .3
        self.frame = None

        print("[INFO] Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        self.frame = image
        blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.layer_names)

    def merge_overlapping_boxes(self, objs):
        box_list = [box for (box, _, _, _) in objs]
        conf_list = [conf for (_, conf, _, _) in objs]
        ids = cv2.dnn.NMSBoxes(box_list, conf_list, self.desired_conf, self.desired_thres)
        if ids.__len__() == objs.__len__():
            return objs
        ids = ids.flatten()
        return [objs[id] for id in ids]

    def extract_objects(self, layer_outputs):

        h, w = self.frame.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in layer_outputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.desired_conf and classID in (self.CLASS_PERSON, self.CLASS_BALL):

                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    if x < 0 or y < 0 or width / self.frame.shape[0] > .5 or height / self.frame.shape[1] > .5:
                        continue

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # return array of elements
        # (bounding_box, pred_confidence, pred_class, valid_box)
        # where valid box => person or soccer ball (to be refined later)
        return [[box, conf, cls, None] for (box, conf, cls) in zip(boxes, confidences, classIDs)]

    def kmeans_determine_team(self, objs, predictor):

        for idx, obj in enumerate(objs):
            (box, _, cls, _) = obj
            if cls == self.CLASS_BALL:
                objs[idx][2] = 3
            elif cls != self.CLASS_PERSON:
                objs[idx][3] = False
            else:
                objs[idx][2] = cc.kmeans_predict_team(self.to_image(box), predictor)
        return objs

    def draw_bounding_boxes(self, image, objs):

        nimage = np.copy(image)

        for (box, conf, cls, text) in objs:
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            color = self.COLORS[cls]
            cv2.rectangle(nimage, (x, y), (x + w, y + h), color, 2)
            cv2.putText(nimage, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return nimage

    def to_image(self, box):
        return self.frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]


def image_preprocess(image):
    lower_color = np.array([35, 100, 60])
    upper_color = np.array([60, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.dilate(mask, np.ones((6, 6), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((50, 50), np.uint8))

    return cv2.bitwise_and(image, image, mask=mask)
