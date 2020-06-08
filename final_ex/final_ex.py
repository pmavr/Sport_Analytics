import numpy as np
import time
import cv2

def show_image(img, msg=''):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


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

# input_file = "cutvideo.mp4"
# input_file = "chelsea_manchester.mp4"
# input_file = "clips/olympiakos_panaitwlikos.mp4"
# team1_mask = ([105, 150, 0], [135, 255, 255])
# team2_mask = ([0, 50, 50], [10, 255, 255])


input_file = "../clips/aris_aek.mp4"
team1_mask = ([25, 125, 125], [35, 255, 255])
team2_mask = ([90, 50, 0], [135, 255, 255])

output_file = "cutvideo_out.avi"
labels_file = "yolov3.txt"
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"
desired_conf = .5
desired_thres = .3

CLASS_PERSON = 0
CLASS_BALL = 32

# load the COCO class labels our YOLO model was trained on
LABELS = open(labels_file).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
yolo = cv2.dnn.readNetFromDarknet(config_file, weights_file)
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

layer_names = yolo.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_file)

success, frame = vs.read()
# frame  = cv2.imread('FA CUP FIELD LINES IV.jpg')
# success = True
writer = None



while success:

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)

    output = yolo.forward(layer_names)

    box_list, conf_list, class_list = extract_objects(frame, output)

    non_overlapping_boxes_IDs = cv2.dnn.NMSBoxes(box_list, conf_list, desired_conf, desired_thres)

    if len(non_overlapping_boxes_IDs) > 0:
        # loop over the indexes we are keeping
        for i in non_overlapping_boxes_IDs.flatten():
            # extract the bounding box coordinates
            (x, y) = (box_list[i][0], box_list[i][1])
            (w, h) = (box_list[i][2], box_list[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[class_list[i]]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_list[i]], conf_list[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Match Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # check if the video writer is None
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter(output_file, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # writer.write(frame)

    success, frame = vs.read()



print("[INFO] cleaning up...")
# writer.release()
vs.release()
cv2.destroyAllWindows()
