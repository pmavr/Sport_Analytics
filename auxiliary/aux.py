import cv2
import numpy as np


class VideoSource:

    def __init__(self):
        self.video_file = "../clips/belgium_japan.mp4"
        self.frame_resolution = (1280, 720)
        self.frame_count = 0
        self.source = cv2.VideoCapture(self.video_file)
        self.continue_playback = True
        # input_file = "../clips/aris_aek.mp4"
        # input_file = "../clips/chelsea_manchester.mp4"

    def get_frame(self):

        success, frame = self.source.read()
        if success and self.continue_playback:
            frame = cv2.resize(frame, self.frame_resolution)
            self.frame_count += 1
        else:
            frame = None
        return frame

    def display_frame(self, frame):
        cv2.imshow('Playing video...', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.continue_playback = False
        if key == ord('p'):
            cv2.waitKey(-1)

    def clean_up(self):
        print("[INFO] cleaning up...")
        self.source.release()
        cv2.destroyAllWindows()



def show_image(img, msg=''):
    """
    Displays an image. Esc char to close window
    :param img: Image to be displayed
    :param msg: Optional message-title for the window
    :return:
    """
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def export_image_as_file(image_to_be_exported, filename):
    path = '../tmp/'
    cv2.imwrite(image_to_be_exported, path+filename)


def remove_white_dots(image, iterations=1):
    # do connected components processing
    for j in range(iterations):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)
        # get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= 100:  # keep
                result[labels == i + 1] = 255

        image = result
        image = cv2.bitwise_not(image, image)

    return result



