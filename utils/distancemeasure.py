import numpy as np
from scipy.spatial import distance
from scipy.stats import chisquare
from scipy.stats import wasserstein_distance

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


def cosine(x, y):
    return distance.cosine(x, y)


def chebyshev(x, y):
    return distance.chebyshev(x, y)


def manhattan(x, y):
    return distance.cityblock(x,y)


def earth_movers(x, y):
    return wasserstein_distance(x, y)


def chi_square(x, y):
    return chisquare(x, y)
