import cv2
from pathlib import Path


def get_project_root() -> Path:
    '''
    :return:  path without slash in the end.
    '''
    return Path(__file__).parent


def get_world_cup_2014_dataset_path():
    return f'{get_project_root()}/datasets/world_cup_2014/'


def get_edge_map_generator_model_path():
    return f'{get_project_root()}/edge_map_generator/generated_models/'


def get_homography_estimator_model_path():
    return f'{get_project_root()}/homography_estimator/generated_models/'

def show_image(img, msg=''):
    """
    Displays an image. Esc char to close window. For debugging purposes
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


def video_player(video_file):
    """Reproduces a video file. For debugging purposes"""
    import time
    input_file = video_file
    vs = cv2.VideoCapture(input_file)
    success, frame = vs.read()
    while success:
        cv2.imshow('Match Detection', frame)
        time.sleep(0.1)
        # writer.write(frame_with_boxes)
        # video play - pause - quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
        success, frame = vs.read()
    vs.release()
    cv2.destroyAllWindows()

