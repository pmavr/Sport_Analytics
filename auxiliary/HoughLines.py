import numpy as np
import cv2
import matplotlib.pyplot as plt
from auxiliary import aux


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
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

def houghLines(image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # # blurredImage = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # edgeImage = cv2.Canny(gray_image, 50, 150)
    # grass_field = extract(image, [36, 25, 25], [86, 255, 255])
    # edgeImage2 = cv2.Canny(grass_field, 50, 120)
    # kernel = np.ones((4, 4), np.uint8)
    # dilation = cv2.dilate(edgeImage2, kernel, iterations=1)

    # Detect points that form a line
    dis_reso = 1  # Distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # Angular resolution in radians of the Hough grid
    threshold = 250  # minimum no of votes

    minLineLength = 30
    maxLineGap = 10
    # houghLines = cv2.HoughLinesP(dilation, dis_reso, theta, threshold, minLineLength, maxLineGap)
    houghLines = cv2.HoughLines(dilation, dis_reso, theta, threshold)

    houghLinesImage = np.zeros_like(image)  # create and empty image

    if houghLines is not None:
        # for i in range(0, len(houghLines)):
        #     l = houghLines[i][0]
        #     cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        tmpa = drawhoughLinesOnImage(image, houghLinesImage)

    # houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_RGB2GRAY)

    houghLines2 = cv2.HoughLinesP(houghLinesImage, dis_reso, theta, 300, minLineLength, maxLineGap)
    houghLinesImage2 = np.zeros_like(image)
    if houghLines2 is not None:
        for i in range(0, len(houghLines2)):
            l = houghLines2[i][0]
            cv2.line(houghLinesImage2, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    orginalImageWithHoughLines = blend_images(houghLinesImage2, image)  # add two images together, using image blending

    return orginalImageWithHoughLines


img = cv2.imread('../clips/frame0.jpg')
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grass_field = extract(img, [40, 60, 60], [60, 255, 255])

_, grass_field = cv2.threshold(grass_field, 70, 255, cv2.THRESH_BINARY_INV)

# edgeImage = cv2.Canny(grass_field, 50, 120)

kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(grass_field, kernel, iterations=3)
erosion = cv2.erode(dilation, kernel, iterations=2)

mask = cv2.bitwise_not(erosion, erosion)

tmp = cv2.Canny(mask, 50, 120)

tmp = cv2.dilate(tmp, kernel, iterations=3)

kernel = np.ones((2, 2), np.uint8)
tmp = cv2.erode(tmp, kernel, iterations=5)

output = houghLines(tmp)



plt.imshow(output), plt.title('dilation2'), plt.show()
