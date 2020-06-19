import numpy as np

def find_intersection(l1, l2):

    # Calculate intercept and gradient of first line
    m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
    b1 = l1[1] - m1 * l1[0]

    # If line is vertical, manually derive intersection
    if l2[0] == l2[2]:
        return np.array([l2[0], m1 * l2[0] + b1])

    # Find intercept and gradient of second line
    m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
    b2 = l2[1] - m2 * l2[0]

    # Calculate intercepts of lines
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return np.array([x, y])

