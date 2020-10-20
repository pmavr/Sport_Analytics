
import cv2
import numpy as np
from auxiliary import HoughLines as hl, aux

def get_line_endpoints(rho, theta):
    line_length = 2200
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + line_length * (-b))
    y1 = int(y0 + line_length * (a))
    x2 = int(x0 - line_length * (-b))
    y2 = int(y0 - line_length * (a))
    return (x1, y1), (x2, y2)

def draw_line(img, rho, theta):
    (x1, y1), (x2, y2) = get_line_endpoints(rho, theta)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    return img

def drawhoughLinesOnImage(image, hl_img, houghLines):
    for (r, th) in houghLines:
        hl_img = draw_line(image, r, th)
    return hl_img

def houghLines(image, threshold=95):
    houghLines = cv2.HoughLines(image,
                                rho=1,                  # Distance resolution of the accumulator in pixels.
                                theta=1*np.pi / 100,    # Angle resolution of the accumulator in radians.
                                threshold=130   # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
                                )
    tmp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    houghLinesImage = np.zeros_like(tmp_image)


    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])
        houghLinesImage = drawhoughLinesOnImage(tmp_image, houghLinesImage, houghLines)


    return houghLines, houghLinesImage

def houghLinesP(image):
    houghLines = cv2.HoughLinesP(image,
                                 rho=1,             # Distance resolution of the accumulator in pixels.
                                 theta=np.pi / 180,   # Angle resolution of the accumulator in radians.
                                 threshold=100,  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
                                 minLineLength=75,  # Minimum line length. Line segments shorter than that are rejected.
                                 maxLineGap=75      # Maximum allowed gap between points on the same line to link them.
                                 )
    tmp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    houghLinesImage = np.zeros_like(tmp_image)

    if houghLines is not None:
        houghLines = houghLines.reshape(houghLines.shape[0], houghLines.shape[2])
        for l in houghLines:
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv2.LINE_AA)

    return houghLines, houghLinesImage

input_file = "../clips/belgium_japan.mp4"
# input_file = "../clips/aris_aek.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
vs = cv2.VideoCapture(input_file)
success, frame = vs.read()

while success:
    img = cv2.resize(frame, (1280, 720))

    img_c = np.copy(img)

    img_c = aux.detect_court(img_c)
    # img_c = aux.detect_white_pixels(img_c)

    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    _, img_c = cv2.threshold(img_c, 150, 200, cv2.THRESH_BINARY)
    img_c = cv2.Canny(img_c, 500, 200)

    lines, final_img = houghLinesP(img_c)

    # show dual images
    concat = np.concatenate((img, final_img), axis=1)
    concat = cv2.resize(concat, (0, 0), None, .75, 1)
    cv2.imshow('Display lines', concat)

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
# writer.release()
cv2.destroyAllWindows()