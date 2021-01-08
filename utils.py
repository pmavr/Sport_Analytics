import cv2
import tensorflow as tf
import torch
from pathlib import Path
import pickle


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


def save_model(model, optimizer, filename):
    model = model.to('cpu')
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


def load_model(model, filename):
    parameters = torch.load(filename)
    model.load_state_dict(parameters['state_dict'])
    return model


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


class Normalize:

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Normalize a tensor with mean and standard deviation. Based pytorch implementation.
            """
        dtype = tensor.dtype
        mean = tf.convert_to_tensor(self.mean, dtype=dtype)
        std = tf.convert_to_tensor(self.std, dtype=dtype)
        if tf.reduce_any(tf.equal(std, 0)):
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = tf.reshape(mean, (-1, 1, 1))
        if std.ndim == 1:
            std = tf.reshape(std, (-1, 1, 1))
        tensor = tf.divide(tf.subtract(tensor, mean), std)
        tensor = tf.reshape(tensor, (tensor.shape[1], tensor.shape[2], tensor.shape[0]))
        return tensor