import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class ColorClusters:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def colorClusters(self):
        # convert to rgb from bgr
        img = cv2.cvtColor(self.IMAGE, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def plotClusters(self):
        # plotting
        fig = plt.figure()
        ax = Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color=rgb_to_hex(self.COLORS[label]))
        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')
        plt.show()


def remove_green(img):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([60, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_colors = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return [hsv_colors[0][0][0], hsv_colors[0][0][1], hsv_colors[0][0][2]]


def train_clustering(imgs, n_clusters=2):
    print('[INFO] Train team predictor...')
    filtered_images = [remove_green(image) for image in imgs]
    clusters = 2
    dominant_colors = []
    for filtered_image in filtered_images:
        dc = ColorClusters(filtered_image, clusters)
        colors = dc.colorClusters()
        for c in colors:
            col = rgb_to_hsv(c[0], c[1], c[2])
            if col[0] > 10 and col[2] > 100:
                dominant_colors.append(col)
    # hsv_colors = [rgb_to_hsv(c[0], c[1], c[2]) for c in dominant_colors]

    # predictor = AgglomerativeClustering(n_clusters=clusters, linkage="average").fit(hsv_colors)
    # predictor = KMeans(n_clusters=n_clusters).fit(hsv_colors)
    predictor = KMeans(n_clusters=n_clusters).fit(dominant_colors)
    return predictor, dominant_colors


def predict_team(image, predictor, n_clusters=2):
    filtered_image = remove_green(image)
    dominant_colors = []

    dc = ColorClusters(filtered_image, n_clusters)
    colors = dc.colorClusters()
    for c in colors:
        col = rgb_to_hsv(c[0], c[1], c[2])
        if col[2] < 30:
            continue
        dominant_colors.append(col)
    # hsv_color = rgb_to_hsv(dominant_colors[0][0], dominant_colors[0][1], dominant_colors[0][2])

    # pred = predictor.predict(np.array(hsv_color).reshape(1, -1))
    pred = predictor.predict(np.array(dominant_colors).reshape(1, -1))
    return np.asscalar(pred)

