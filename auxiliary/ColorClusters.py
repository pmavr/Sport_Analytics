import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from auxiliary.aux import show_image


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
        # img = cv2.cvtColor(self.IMAGE, cv2.COLOR_BGR2RGB)
        img = self.IMAGE
        # reshaping to a list of pixels
        # img = img.reshape((img.shape[0] * img.shape[1], 3))

        # using k-means to cluster pixels
        if (img.shape[0]==0):
            print()
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

    def plotHistogram(self):

        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()


def remove_green(img):
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([60, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask, mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    # aux.show_image(res)
    return res


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    hsv_colors = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return [hsv_colors[0][0][0], hsv_colors[0][0][1], hsv_colors[0][0][2]]


def train_clustering(imgs, n_clusters=2):
    print('[INFO] Train team predictor...')

    dominant_colors = [find_dominant_color(img) for img in imgs]
    dominant_colors = [np.array([c[0][0] + c[1][0]]) for c in dominant_colors]
    predictor = KMeans(n_clusters=n_clusters).fit(dominant_colors)
    return predictor


def find_dominant_color(img, n_dominant_colors=2):
    im = remove_green(img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    color_mask = np.any(im != [0, 0, 0], axis=-1)
    im = im[color_mask]
    dc = ColorClusters(im, n_dominant_colors)
    colors = dc.colorClusters()
    hsv_colors = [rgb_to_hsv(c[0], c[1], c[2]) for c in colors]
    return hsv_colors


# def predict_team(imgs, predictor):
#     dominant_colors = [find_dominant_color(img) for img in imgs]
#     dominant_colors = [np.array([c[0][0] + c[1][0]]) for c in dominant_colors]
#     return predictor.predict(dominant_colors)

def predict_team(img, predictor):
    dominant_color = find_dominant_color(img)
    dominant_color = np.array([[dominant_color[0][0] + dominant_color[1][0]]]).astype('uint16')
    pred = predictor.predict(dominant_color)
    return np.asscalar(pred)
