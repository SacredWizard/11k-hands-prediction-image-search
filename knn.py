import numpy as np
import scipy

# Euclidean distance
def euclidean(x, y):
    dist = scipy.spatial.distance.cdist(x, y, 'euclidean')
    return dist


class KNN(object):
    def __init__(self, k):
        self.k_value = k
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Make a classification prediction with neighbors
    def predict(self, test):
        predictions = []
        dist_matrix = euclidean(test, self.X_train)
        for row in dist_matrix:
            neighbor_indices = np.array(row).argsort()[:self.k_value]
            output_values = [self.y_train[i] for i in neighbor_indices]
            prediction = max(set(output_values), key=output_values.count)
            predictions.append(prediction)
        return predictions

