import time

import numpy as np
import pandas as pd
import pymongo
import pymongo.errors
from sklearn.decomposition import NMF

from classes import global_connections, distancemeasure, global_constants

constants = global_constants.GlobalConstants()
connection = global_connections.GlobalConnections()


def nmf(data):
    if data is not None:
        model = NMF(n_components=5, beta_loss='frobenius', init='nndsvd', random_state=0)
        print(model)
        W = model.fit_transform(data)
        H = model.components_
        tt1 = time.time()
        for i in range(H.shape[0]):
            print("Latent Feature: {}\n{}".format(i + 1, sorted(((i, v) for i, v in enumerate(H[i])),
                                                                key=lambda x: x[1], reverse=True)))

        print("Time NMF {}".format(time.time() - tt1))
        return W
    pass


def sort_print_pandas(data):
    tt = time.time()
    df = pd.DataFrame(data)
    df = df.apply(func=lambda y: sorted(((i, v) for i, v in enumerate(y)), key=lambda x: x[1], reverse=True), axis=1)
    print(df)
    print("Time Pandas {}".format(time.time() - tt))


def get_object_feature_matrix(extractor):
    try:
        vector_list = []
        cursor = connection.mongo_client.features[extractor.lower()].find({}, {'_id': 0})
        if extractor == 'LBP':
            for rec in cursor:
                vector_list.append(rec['featureVector'].split(','))
            return np.array(vector_list).astype(np.float)

        for rec in cursor:
            vector_list.append(rec['featureVector'])
        return np.array(vector_list)

    except pymongo.errors.ServerSelectionTimeoutError as e:
        print("Timeout:\n{}".format(e))
    except Exception as e:
        print("Exception occurred:\n{}".format(e))


def get_object_feature_matrix_pandas(type):
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    cursor = mongo_client.features[type.lower()].find({}, {'_id': 0})
    df = pd.DataFrame(list(cursor))
    df.set_index('ImageId', inplace=True)
    return df.featureVector.apply(pd.Series)


def extract_reduce(extract, reduce):
    if reduce in [constants.LDA, constants.PCA, constants.NMF, constants.SVD] and \
            extract in [constants.CM, constants.HOG, constants.LBP, constants.SIFT]:
        return globals()[reduce.lower()](get_object_feature_matrix(extract))


def task1():
    dimension_reduction_method = 'NMF'
    feature_extractor = 'HOG'
    return extract_reduce(feature_extractor, dimension_reduction_method)


def task2():
    distances = distancemeasure.DistanceMeasure()
    print("Task 2")
    data = task1()
    dimension_reduction_method = 'NMF'
    feature_extractor = 'HOG'
    image = 'Hand_0000025.jpg'
    cursor = connection.mongo_client.features[feature_extractor.lower()].find()

    vals = {}
    for cur, row in enumerate(cursor):
        vals[cur] = row['ImageId']
    scores = []
    for row in data:
        scores.append(distances.nvsc1(data[0], row))
    scores_euclidian = []
    for row in data:
        scores_euclidian.append(distances.euclidian(data[0], row))
    indexes = np.argsort(scores)[::-1]
    for i in indexes:
        print("Image: {}, Score: {}".format(vals[i], scores[i]*100))


def main():
    t1 = time.time()
    # task1()
    task2()
    print(time.time() - t1)


if __name__ == '__main__':
    main()
