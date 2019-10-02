import numpy as np


class DistanceMeasure:

    def __init__(self):
        pass

    def nvsc1(self, x, y):
        return np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

    def euclidian(self, x, y):
        return np.linalg.norm(x - y)
