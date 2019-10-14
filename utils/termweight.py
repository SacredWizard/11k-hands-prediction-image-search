from numpy import argsort
import numpy as np


def get_data_latent_semantics(data_m, k, images):
    term = [argsort(-data_m[:, i]) for i in range(k)]
    tw = []
    data_m = np.asarray(data_m)
    # for each latent semantic (k)
    for i in range(k):
        k_tw = []
        # for each data object
        for j in range(len(term[i])):
            k_tw.append([images[term[i][j]],data_m[term[i][j],i]])
        tw.append(k_tw)
    return tw


def get_feature_latent_semantics(feature_m, k):
    term = [argsort(-feature_m[i, :]) for i in range(k)]
    tw = []
    feature_m = np.asarray(feature_m)
    # for each latent semantic (k)
    for i in range(k):
        k_tw = []
        # for each feature
        for j in range(len(term[i])):
            k_tw.append([term[i][j],feature_m[i,term[i][j]]])
        tw.append(k_tw)
    return tw


def print_tw(data_m, feature_m):
    """ prints Term Weight Pairs to console"""
    images = data_m['imageId'].tolist()
    data_m = np.array(data_m['reducedDimensions'].tolist())
    data_tw = get_data_latent_semantics(data_m, data_m.shape[1], images)

    feature_tw = get_feature_latent_semantics(feature_m, feature_m.shape[0])

    separator = "\n{}\n".format("-" * 200)

    print(separator)
    print("Data Latent Semantics")
    print(separator)
    for i in range(len(data_tw)):
        print("LS {}  -->  {}\n".format(i + 1, data_tw[i]))

    print(separator)
    print("Feature Latent Semantics")
    print(separator)
    for i in range(len(feature_tw)):
        print("LS {}  -->  {}\n".format(i + 1, feature_tw[i]))