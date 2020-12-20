from pathlib import Path


def get_project_root() -> Path:
    '''
    :return:  path without slash in the end.
    '''
    return Path(__file__).parent
