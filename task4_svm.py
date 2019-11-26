import numpy as np
import json
import os
from pandas.io.json import json_normalize
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from phase3.task1 import compute_latent_semantic_for_label, reduced_dimensions_for_unlabelled_folder
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import random

csv_reader = CSVReader()
             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        # initializing matrix of zeros and size of training data
        K = np.zeros((n_samples, n_samples))

        # getting polynomial kernel for each sample and storing in K
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":   

    fea_ext_mod = "HOG"
    dim_red_mod = "PCA"
    dist_func = "euclidean"
    k_value = 20
    training_set = 'C:\\Users\\baani\OneDrive\Documents\Arizona State University\Fall 2019\CSE 515\Project\PhaseIII\CSE515\Dataset3\Labelled\Set1'
    test_set = 'C:\\Users\\baani\OneDrive\Documents\Arizona State University\Fall 2019\CSE 515\Project\PhaseIII\CSE515\Dataset3\\Unlabelled\Set 1'
    label = "dorsal"
    obj_lat,feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod, 
                                        dim_red_mod, label , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label, str(k_value))
    csv_reader.save_to_csv(obj_lat, feat_lat, filename)

    label_p = 'palmar'
    obj_lat_p,feat_lat_p, model_p = compute_latent_semantic_for_label(fea_ext_mod, 
                                        dim_red_mod, label_p , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label_p, str(k_value))
    csv_reader.save_to_csv(obj_lat_p, feat_lat_p, filename)
    
    x_train = obj_lat['reducedDimensions'].tolist()
    x_train += (obj_lat_p['reducedDimensions'].tolist())
    red_dim_unlabelled_images = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label, training_set, test_set)
    x_test = red_dim_unlabelled_images['reducedDimensions'].tolist()

    dim_red = DimensionReduction(fea_ext_mod,dim_red_mod,k_value)
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    y_train = [i.split(' ')[0] for i in labelled_aspect]

    labelled_aspect = dim_red.get_metadata("imageName", obj_lat_p['imageId'].tolist())['aspectOfHand'].tolist()
    y_train += ([i.split(' ')[0] for i in labelled_aspect])
    
    unlabelled_aspect = dim_red.get_metadata("imageName", red_dim_unlabelled_images['imageId'].tolist())['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]

    # makes into arrays and transforms the training labels into 1 for "dorsal", -1 for "palmar" data points 
    x_train = np.array(x_train)
    y_train = list(map(lambda x:1 if x=="dorsal" else -1,y_train))
    y_train = np.array(y_train)

    # shuffling the training data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    print(y_train)
    x_test = np.array(x_test)

    clf = SVM(polynomial_kernel, C=500)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    print("Predictions:")
    print(predictions)

    # transforms the testing labels into 1 for "dorsal", -1 for "palmar" data points 
    y_test = list(map(lambda x:1 if x=="dorsal" else -1,y_test))
    print("Expected Output:")
    print(y_test)
    correct = np.sum(predictions == y_test)
    print("%d out of %d predictions correct" % (correct, len(predictions)))