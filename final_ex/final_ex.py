
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

# Draw the lines represented in the hough accumulator on the original image
def drawhoughLinesOnImage(image, houghLines):
    for line in houghLines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Different weights are added to the image to give a feeling of blending
def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta, gamma)

def houghLines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    grass_field = extract(image, [35, 100, 100], [45, 255, 255])
    edgeImage2 = cv2.Canny(grass_field, 50, 120)
    # Detect points that form a line
    dis_reso = 1  # Distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # Angular resolution in radians of the Hough grid
    threshold = 170  # minimum no of votes
    houghLines = cv2.HoughLines(edgeImage2, dis_reso, theta, threshold)
    houghLinesImage = np.zeros_like(image)  # create and empty image
    drawhoughLinesOnImage(houghLinesImage, houghLines)  # draw the lines on the empty image
    orginalImageWithHoughLines = blend_images(houghLinesImage, image)  # add two images together, using image blending
    return edgeImage2, orginalImageWithHoughLines

def extract(img, lower_range, upper_range):
    lower_color = np.array(lower_range)
    upper_color = np.array(upper_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def color_intensity(img, color_mask):
    res = extract(img, color_mask[0], color_mask[1])
    return cv2.countNonZero(res)

def get_player_color(image, color1_mask, color2_mask):

    # scale_percent = 300  # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # show_image(resized)

    grass_pixels = color_intensity(image, ([45, 40, 40], [50, 255, 255]))

    team1_pixels = color_intensity(image, color1_mask)   #  aris

    team2_pixels = color_intensity(image, color2_mask)   # aek

    if team1_pixels > team2_pixels and team1_pixels >= 50 and grass_pixels >= 50:
        return color1_mask[1], 'aris'
    if team2_pixels > team1_pixels and team2_pixels >= 50 and grass_pixels >= 50:
        return color2_mask[1], 'aek'
    else:
        return 0, 'none'

# input_file = "cutvideo.mp4"
# input_file = "chelsea_manchester.mp4"
# input_file = "clips/olympiakos_panaitwlikos.mp4"
# team1_mask = ([105, 150, 0], [135, 255, 255])
# team2_mask = ([0, 50, 50], [10, 255, 255])


input_file = "clips/aris_aek.mp4"
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
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_file)
total_frames = 0
success, frame = vs.read()
# frame  = cv2.imread('FA CUP FIELD LINES IV.jpg')
# success = True
writer = None



while success:
    total_frames += 1
    # edge_img, img_with_hough_lines = houghLines(frame)
    if total_frames == 300:
        print(total_frames)
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

                clr, lbl = get_player_color(player_img, team1_mask, team2_mask)

                if clr != 0:
                    boxes.append([x, y, int(width), int(height), clr, lbl])
                    # boxes.append([x, y, int(width), int(height)])
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
            # color = [int(c) for c in COLORS[classIDs[i]]]
            color = boxes[i][4]
            label = boxes[i][5]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}: {:.4f}".format(label, confidences[i])
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
print(total_frames)