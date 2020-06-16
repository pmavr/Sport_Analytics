import numpy as np
import cv2
from matplotlib import pyplot as plt
from auxiliary.HoughLines import extract
from auxiliary.ColorClusters import remove_green

img = cv2.imread('../clips/frame0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# kernel = np.ones((5, 5), np.float32) / 25
# dst = cv.filter2D(img, -1, kernel)



# kernel = np.ones((3, 3), np.uint8)
#
# dilation = cv2.dilate(img, kernel, iterations=8)
#
# erosion = cv2.erode(dilation, kernel, iterations=6)

grass_field = extract(img, [55, 40, 40], [86, 255, 255])
dilation2 = cv2.dilate(grass_field, kernel, iterations=3)
edge = cv2.Canny(dilation2, 50, 150)

# # Detect points that form a line
# dis_reso = 1  # Distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # Angular resolution in radians of the Hough grid
# threshold = 650  # minimum no of votes
#
# minLineLength = 30
# maxLineGap = 10
# houghLines = cv2.HoughLinesP(edge, dis_reso, theta, threshold, minLineLength, maxLineGap)
#
#
# houghLinesImage = np.zeros_like(img)  # create and empty image
#
# if houghLines is not None:
#     for i in range(0, len(houghLines)):
#         l = houghLines[i][0]
#         cv2.line(houghLinesImage, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)
#
# houghLinesImage = cv2.cvtColor(houghLinesImage, cv2.COLOR_RGB2GRAY)
#


plt.subplot(121)
plt.imshow(dilation2), plt.title('dilation2')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(edge), plt.title('grass_field')
plt.xticks([]), plt.yticks([])
plt.show()
