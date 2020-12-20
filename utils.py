from pathlib import Path


def get_project_root() -> Path:
    '''
    :return:  path without slash in the end.
    '''
    return Path(__file__).parent


def get_world_cup_2014_dataset_path():
    return f'{get_project_root()}datasets/original/world_cup_2014/'


def get_edge_map_generator_dataset_path():
    return f'{get_project_root()}datasets/generated/edge_map_generator/'


def get_grass_mask_estimator_dataset_path():
    return f'{get_project_root()}datasets/generated/grass_mask_estimator/'


def get_homography_estimator_dataset_path():
    return f'{get_project_root()}datasets/generated/homography_estimator/'


def get_edge_map_generator_model_path():
    return f'{get_project_root()}edge_map_generator/generated_models/'
