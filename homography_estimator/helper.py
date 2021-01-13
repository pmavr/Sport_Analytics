import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from homography_estimator.Camera import Camera


def infer_features_from_edge_map(
        model,
        edge_map,
        transform):

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        cudnn.benchmark = True

    model.eval()
    with torch.no_grad():
        x = transform(edge_map)
        x = torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
        x = x.to(device)
        feat = model.feature_numpy(x)

    return feat


def camera_to_edge_map(binary_court, camera_params, img_h=720, img_w=1280):
    points = binary_court['points']
    line_segment_indexes = binary_court['line_segment_index']

    camera = Camera(camera_params)
    edge_map = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    n = line_segment_indexes.shape[0]
    for i in range(n):
        idx1, idx2 = line_segment_indexes[i][0], line_segment_indexes[i][1]
        p1, p2 = points[idx1], points[idx2]
        q1 = camera.project_point_on_frame(p1[0], p1[1])
        q2 = camera.project_point_on_frame(p2[0], p2[1])
        cv2.line(edge_map, tuple(q1), tuple(q2), color=(255, 255, 255), thickness=4)

    return edge_map