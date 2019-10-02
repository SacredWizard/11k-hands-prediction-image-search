import time

import numpy as np
import pandas as pd
import pymongo
from sklearn.decomposition import NMF

import global_constants

constants = global_constants.GlobalConstants()


def nmf(data):
    model = NMF(n_components=5, beta_loss='frobenius', init='nndsvd', random_state=0)
    print(model)
    print(data.shape)
    W = model.fit_transform(data)
    H = model.components_
    print(W.shape)
    print(H.shape)
    tt1 = time.time()
    for l in range(H.shape[0]):
        print("Latent Feature: {}\n{}".format(l + 1, sorted(((i, v) for i, v in enumerate(H[l])),
                                                            key=lambda x: x[1], reverse=True)))
    print("Time NP {}".format(time.time() - tt1))
    pass


def sort_print_pandas(data):
    tt = time.time()
    df = pd.DataFrame(data)
    df = df.apply(func=lambda y: sorted(((i, v) for i, v in enumerate(y)), key=lambda x: x[1], reverse=True), axis=1)
    print(df)
    print("Time Pandas {}".format(time.time() - tt))


def get_object_feature_matrix(extractor):
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    vector_list = []
    cursor = mongo_client.features[extractor.lower()].find({}, {'_id': 0})

    if extractor == 'LBP':
        for rec in cursor:
            vector_list.append(rec['featureVector'].split(','))
        return np.array(vector_list).astype(np.float)

    for rec in cursor:
        vector_list.append(rec['featureVector'])

    return np.array(vector_list)
    pass


def get_object_feature_matrix_pandas(type):
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    cursor = mongo_client.features[type.lower()].find({}, {'_id': 0})
    df = pd.DataFrame(list(cursor))
    df.set_index('ImageId', inplace=True)
    return df.featureVector.apply(pd.Series)


def extract_reduce(extract, reduce):
    if reduce in [constants.LDA, constants.PCA, constants.NMF, constants.SVD] and \
            extract in [constants.CM, constants.HOG, constants.LBP, constants.SIFT]:
        globals()[reduce.lower()](get_object_feature_matrix(extract))


def task1():
    dimension_reduction_method = 'NMF'
    feature_extractor = 'LBP'
    extract_reduce(feature_extractor, dimension_reduction_method)


def main():
    task1()


if __name__ == '__main__':
    main()
