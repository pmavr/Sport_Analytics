import cv2, \
    numpy as np
from auxiliary import ColorClusters as cc,\
    ObjectDetector as od, \
    HoughLines as hl


def object_detector_pipeline(image):
    frame_o = np.copy(image)
    img_for_object_detector = od.image_preprocess(frame_o)

    output = yolo.predict(img_for_object_detector)
    objects = yolo.extract_objects(output)

    refined_objects = yolo.merge_overlapping_boxes(objects)

    final_objects = yolo.kmeans_determine_team(refined_objects, team_predictor)
    return  final_objects


def court_detector_pipeline(image):
    frame_c = np.copy(image)
    img_c = hl.image_preprocess(frame_c)
    lines, img_with_hough_lines = hl.houghLines(img_c, frame_c)

    hor_lines = []
    ver_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line
            if hl.is_horizontal(theta):
                hor_lines.append(line)
            elif hl.is_vertical(theta):
                ver_lines.append(line)

    ref_hor_lines = hl.refine_lines(hor_lines, rtol=.125)
    ref_ver_lines = hl.refine_lines(ver_lines, rtol=.125)

    lines = []
    for line in ref_hor_lines:
        lines.append(line)
    for line in ref_ver_lines:
        lines.append(line)

    return hl.get_intersection_points(frame, lines)


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

    points = court_detector_pipeline(frame_resized)

    identified_objects = object_detector_pipeline(frame_resized)

    frame_with_boxes = yolo.draw_bounding_boxes(frame_resized, identified_objects)

    for p in points:
        if p is not None:
            cv2.line(frame_with_boxes, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (255, 255, 0), 10)

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
