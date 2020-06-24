import numpy as np
import cv2
from auxiliary.aux import show_image
from auxiliary.HoughLines import image_preprocess, is_horizontal, is_vertical, refine_lines, houghLines, get_court_intersection_points, get_intersection_points, \
    blend_images


frame = cv2.imread('../clips/frame4.jpg')
frame_resized = cv2.resize(frame, (1280, 720))

img = image_preprocess(frame_resized)

lines, image_with_lines = houghLines(img, frame_resized)

hor_lines = list()
ver_lines = list()

for line in lines:
    rho, theta = line
    if is_horizontal(theta):
        hor_lines.append(line)
    elif is_vertical(theta):
        ver_lines.append(line)

ref_ver_lines = refine_lines(ver_lines, rtol=.125)
ref_hor_lines = refine_lines(hor_lines, rtol=.125)

# if ref_hor_lines is not None:
#     drawhoughLinesOnImage(frame, ref_hor_lines)
# if ref_ver_lines is not None:
#     drawhoughLinesOnImage(frame, ref_ver_lines)

# aux.show_image(frame)
lines = []
for line in ref_hor_lines:
    lines.append(line)
for line in ref_ver_lines:
    lines.append(line)

intersection_points = get_intersection_points(lines)
intersection_points = [p for p in intersection_points if p is not None and p[0]>=0 and p[1]>=0]

for p in intersection_points:
    cv2.line(frame_resized, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (255, 255, 0), 10)

court_intersection_points = get_court_intersection_points()


court_image = cv2.imread('../clips/court.jpg')

src = np.array([[914., 0.], [1091., 0.], [914., 557.7078881563991], [1091., 129.50011366219587]], np.float32)

dst = np.array([[459.0645161112684, 143.2217852219327], [765.7319784379197, 121.76624489962595],
                [1110.6788355551557, 433.11638665653606], [899.7333270184508, 162.6960601898524]], np.float32)

homography_matrix = cv2.getPerspectiveTransform(src, dst)

im_out = cv2.warpPerspective(court_image, homography_matrix, (frame_resized.shape[1], frame_resized.shape[0]))

im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(im_out, 10, 255, cv2.THRESH_BINARY)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

final_image = blend_images(mask, frame_resized)

show_image(final_image)

