import cvxopt as optimizer
import cvxopt.solvers as solver
import numpy as np
from numpy import linalg
import os
from classes.dimensionreduction import DimensionReduction
from utils.inputhelper import get_input_folder
from phase3.task1 import compute_latent_semantic_for_label, reduced_dimensions_for_unlabelled_folder
from utils.excelcsv import CSVReader

csv_reader = CSVReader()


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SupportVectorMachine(object):

    # initializing values
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        number_samples, number_features = X.shape

        # Gram matrix
        # initializing matrix of zeros and size of training data
        K = np.zeros((number_samples, number_samples))

        # getting polynomial kernel for each sample and storing in K
        for i in range(number_samples):
            for j in range(number_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # G and A are sparse matrices
        # P is a square dense or sparse real matrix, which represents a positive semidefinite symmetric matrix
        # q is a real single-column dense matrix
        # h and b are real-single column dense matrices
        # G and A are real dense or sparse matrices

        P = optimizer.matrix(np.outer(y, y) * K)
        q = optimizer.matrix(np.ones(number_samples) * -1)
        A = optimizer.matrix(y, (1, number_samples), 'd')
        b = optimizer.matrix(0.0)

        if self.C is None:
            G = optimizer.matrix(np.diag(np.ones(number_samples) * -1))
            h = optimizer.matrix(np.zeros(number_samples))
        else:
            tmp1 = np.diag(np.ones(number_samples) * -1)
            tmp2 = np.identity(number_samples)
            G = optimizer.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(number_samples)
            tmp2 = np.ones(number_samples) * self.C
            h = optimizer.matrix(np.hstack((tmp1, tmp2)))

        # solves quadratic programming problem
        solution = solver.qp(P, q, G, h, A, b)

        # calculates Lagrange multipliers
        a = np.ravel(solution['x'])

        # support vectors have non zero lagrange multipliers
        support_vectors = a > 1e-5
        ind = np.arange(len(a))[support_vectors]
        self.a = a[support_vectors]
        self.support_vectors = X[support_vectors]
        self.support_vectors_y = y[support_vectors]
        print("---------------------------")
        print("Support Vectors: " + str(len(self.a)))

        # calculates b intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.support_vectors_y[n]
            self.b -= np.sum(self.a * self.support_vectors_y * K[ind[n], support_vectors])
        if len(self.a) > 0:
            self.b /= len(self.a)
        # calculates the weights vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(number_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.support_vectors_y[n] * self.support_vectors[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, support_vectors_y, support_vectors in zip(self.a, self.support_vectors_y, self.support_vectors):
                    s += a * support_vectors_y * self.kernel(X[i], support_vectors)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


def main():
    fea_ext_mod = "HOG"
    dim_red_mod = "PCA"
    dist_func = "euclidean"
    k_value = 30

    training_set = os.path.abspath(get_input_folder("Labelled"))
    test_set = os.path.abspath(get_input_folder("Classify"))
    # training_set = os.path.abspath('Dataset3\Labelled\Set1')
    # test_set = os.path.abspath('Dataset3\\Unlabelled\Set 1')
    label = "dorsal"
    obj_lat, feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod,
                                                                 dim_red_mod, label, k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label, str(k_value))
    csv_reader.save_to_csv(obj_lat, feat_lat, filename)

    label_p = 'palmar'
    obj_lat_p, feat_lat_p, model_p = compute_latent_semantic_for_label(fea_ext_mod,
                                                                       dim_red_mod, label_p, k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label_p, str(k_value))
    csv_reader.save_to_csv(obj_lat_p, feat_lat_p, filename)

    x_train = obj_lat['reducedDimensions'].tolist()
    x_train += (obj_lat_p['reducedDimensions'].tolist())
    red_dim_unlabelled_images = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label,
                                                                         training_set, test_set)
    x_test = red_dim_unlabelled_images['reducedDimensions'].tolist()

    dim_red = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    y_train = [i.split(' ')[0] for i in labelled_aspect]

    labelled_aspect = dim_red.get_metadata("imageName", obj_lat_p['imageId'].tolist())['aspectOfHand'].tolist()
    y_train += ([i.split(' ')[0] for i in labelled_aspect])

    unlabelled_aspect = dim_red.get_metadata("imageName", red_dim_unlabelled_images['imageId'].tolist())['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]

    # makes into arrays and transforms the training labels into 1 for "dorsal", -1 for "palmar" data points 
    x_train = np.array(x_train)
    y_train = list(map(lambda x: 1 if x == "dorsal" else -1, y_train))
    y_train = np.array(y_train)

    # shuffling the training data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_test = np.array(x_test)

    # creates the SVM classifier
    clf = SupportVectorMachine(gaussian_kernel, C=500)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    # transforms the testing labels into 1 for "dorsal", -1 for "palmar" data points 
    y_test = list(map(lambda x: 1 if x == "dorsal" else -1, y_test))

    # calculates and prints the results onto the console
    correct = np.sum(predictions == y_test)
    print("---------------------------")
    accuracy = (correct / len(predictions)) * 100
    print("Accuracy: " + str(accuracy) + "%")
    unlabelled_images = red_dim_unlabelled_images['imageId']
    predicted_labels = list(map(lambda x: "dorsal" if x == 1 else "palmar", predictions))
    actual_labels = list(map(lambda x: "dorsal" if x == 1 else "palmar", y_test))
    print("---------------------------")
    print("Results:")
    print("Image ID, Prediction, Actual")
    for image_id, p, a in zip(unlabelled_images, predicted_labels, actual_labels):
        print("(" + image_id + ", " + p + ", " + a + ")")


if __name__ == "__main__":
    main()