from numpy import argsort
import numpy as np

metadata_list = ['male', 'female', 'dorsal', 'palmar', 'accessories', 'without accessories', 'left', 'right']


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


def get_feature_latent_semantics(feature_m, k, image_metadata=False):
    term = [argsort(-feature_m[i, :]) for i in range(k)]
    tw = []
    feature_m = np.asarray(feature_m)
    # for each latent semantic (k)
    for i in range(k):
        k_tw = []
        # for each feature
        for j in range(len(term[i])):
            if image_metadata:
                k_tw.append([metadata_list[term[i][j]], feature_m[i, term[i][j]]])
            else:
                k_tw.append([term[i][j],feature_m[i,term[i][j]]])
        tw.append(k_tw)
    return tw


def print_tw(data_m, feature_m, image_metadata=False, subject_subject=False):
    """ prints Term Weight Pairs to console"""
    images = data_m['imageId'].tolist()
    data_m = np.array(data_m['reducedDimensions'].tolist())
    data_tw = get_data_latent_semantics(data_m, data_m.shape[1], images)

    if not subject_subject:
        feature_tw = get_feature_latent_semantics(feature_m, feature_m.shape[0], image_metadata)

    separator = "\n{}\n".format("-" * 200)

    print(separator)
    if image_metadata:
        print("Latent Semantics in Image space")
    elif subject_subject:
        print("Top-k Latent Semantics")
    else:
        print("Data Latent Semantics")
    print(separator)
    for i in range(len(data_tw)):
        print("LS {}  -->  {}\n".format(i + 1, data_tw[i]))

    print(separator)
    if not subject_subject:
        if image_metadata:
            print("Latent Semantics in Metadata space")
        else:
            print("Feature Latent Semantics")
        print(separator)
        for i in range(len(feature_tw)):
            print("LS {}  -->  {}\n".format(i + 1, feature_tw[i]))