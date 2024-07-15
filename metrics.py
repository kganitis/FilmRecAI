import numpy as np


def euclidean_distance(u, v):
    return np.dot(u - v, u - v)


def euclidean_distance_custom(u, v):
    """ Euclidean distance between non-zero elements of u and v. """
    diff = np.where((u == 0) | (v == 0), 0, u - v)
    dist = np.sum(diff ** 2)
    return np.sqrt(dist)