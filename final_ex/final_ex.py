
import numpy as np
import time
import cv2


def show_image(msg, img):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)

def color_intensity(img, lower_range, upper_range):
    lower_color = np.array(lower_range)
    upper_color = np.array(upper_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(res)

def get_player_color(image, box):

    # show_image('',image)

    green_pixels = color_intensity(image, [45, 40, 40], [50, 255, 255])

    blue_pixels = color_intensity(image, [105, 10, 0], [135, 255, 255])

    red_pixels = color_intensity(image, [0, 50, 50], [10, 255, 255])

    if blue_pixels >= 50 and green_pixels >= 50:
        return 1
    if red_pixels >= 50 and green_pixels >= 50:
        return 2
    else:
        return 0

input_file = "cutvideo.mp4"
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
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_file)

success, frame = vs.read()
writer = None

while success:

    H, W = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > desired_conf:

                if classID != CLASS_PERSON and classID != CLASS_BALL:
                    continue

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if x < 0 or y < 0:
                    continue

                player_img = frame[y:y + int(height), x:x + int(width)]

                color = get_player_color(player_img, box)

                if color != 0:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, desired_conf,
                            desired_thres)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Match Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, frame = vs.read()

    # check if the video writer is None
#     if writer is None:
#         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#         writer = cv2.VideoWriter(output_file, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
#     writer.write(frame)
#
# writer.release()

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
