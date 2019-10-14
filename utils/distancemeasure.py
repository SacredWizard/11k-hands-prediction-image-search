import numpy as np


def euclidean(x, y):
    """
    Calculates the Euclidean distance between 2 vectors
    :param x: Vector 1
    :param y: Vector 2
    :return: distance between them
    """
    return np.linalg.norm(x - y)


def nvsc1(x, y):
    return np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

