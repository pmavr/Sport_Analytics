
import numpy as np
import cv2


def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta, gamma)


def extract(img, lower_range, upper_range):
    lower_color = np.array(lower_range)
    upper_color = np.array(upper_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def houghLines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blurredImage = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edgeImage = cv2.Canny(gray_image, 50, 150)
    grass_field = extract(image, [36, 25, 25], [86, 255, 255])
    edgeImage2 = cv2.Canny(grass_field, 50, 120)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(edgeImage2, kernel, iterations=1)


    # Detect points that form a line
    dis_reso = 1  # Distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # Angular resolution in radians of the Hough grid
    threshold = 650  # minimum no of votes

    minLineLength = 30
    maxLineGap = 10
    houghLines = cv2.HoughLinesP(dilation, dis_reso, theta, threshold, minLineLength, maxLineGap)


    houghLinesImage = np.zeros_like(image)  # create and empty image

    if houghLines is not None:
        for i in range(0, len(houghLines)):
            l = houghLines[i][0]
            cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

    houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_RGB2GRAY)

    houghLines2 = cv2.HoughLinesP(houghLinesImage, dis_reso, theta, 300, minLineLength, maxLineGap)
    houghLinesImage2 = np.zeros_like(image)
    if houghLines2 is not None:
        for i in range(0, len(houghLines2)):
            l = houghLines2[i][0]
            cv2.line(houghLinesImage2, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    orginalImageWithHoughLines = blend_images(houghLinesImage2, image) # add two images together, using image blending

    return orginalImageWithHoughLines



# img_with_hough_lines = hl.houghLines(frame)
#
# cv2.imshow('Match Detection', img_with_hough_lines)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break