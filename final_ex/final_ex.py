
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

def get_color_mask(hsv):

    # lower mask (0-10)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_color, upper_color)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_color, upper_color)

    # join my masks
    return mask0 + mask1

def get_player_color(image, box):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # show_image('',image)

    # If player has blue jersey
    # blue range
    lower_red = np.array([105, 100, 10])
    upper_red = np.array([135, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    res1 = cv2.bitwise_and(image, image, mask=mask0)
    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    nzCountblue = cv2.countNonZero(res1)

    # If player has red jersy
    # lower red mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper red mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask2 = mask0 + mask1
    res2 = cv2.bitwise_and(image, image, mask=mask2)
    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    nzCountred = cv2.countNonZero(res2)

    if nzCountblue >=40:
        return 1
    if nzCountred >= 40:
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
    # time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, frame = vs.read()




print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
