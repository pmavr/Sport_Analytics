import cv2
from pathlib import Path


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


def get_project_root() -> Path:
    '''
    :return:  path without slash in the end.
    '''
    return Path(__file__).parent


def get_world_cup_2014_dataset_path():
    return f'{get_project_root()}/datasets/world_cup_2014/'


def get_edge_map_generator_model_path():
    return f'{get_project_root()}/edge_map_generator/generated_models/'
