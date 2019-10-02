import numpy as np


def euclidian(x, y):
    return np.linalg.norm(x - y)


def nvsc1(x, y):
    return np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

