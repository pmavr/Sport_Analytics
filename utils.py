import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np


def get_project_root():
    '''
    :return:  path without slash in the end.
    '''
    return f'{Path(__file__).parent}/'


def get_world_cup_2014_dataset_path():
    return f'{get_project_root()}datasets/world_cup_2014/'


def get_grass_mask_estimator_model_path():
    return f'{get_project_root()}grass_mask_estimator/generated_models/'


def get_edge_map_generator_model_path():
    return f'{get_project_root()}edge_map_generator/generated_models/'


def get_homography_estimator_model_path():
    return f'{get_project_root()}homography_estimator/generated_models/'


def save_model(model_components, history, filename):
    """Save trained model along with its optimizer and training, plottable history."""
    state = {}
    for key in model_components.keys():
        component = model_components[key]
        state[key] = component.state_dict()
    state['history'] = history
    torch.save(state, filename)


def load_pickle_file(filename):
    data = []
    with open(filename, 'rb') as file:
        while True:
            try:
                data.append(pickle.load(file))
            except EOFError:
                break
        file.close()
        return data


def save_to_pickle_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()


def plot_image(img, title=''):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]

    for i in range(len(img_list)):
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
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

