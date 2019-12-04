import os
import time

import numpy as np

from classes.LSH import LSH
from classes.dimensionreduction import DimensionReduction
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper
from utils.imageviewer import show_images
from utils.model import Model


def save_model_file():
    constants = GlobalConstants()
    results = MongoWrapper(constants.Mongo().DB_NAME).find(constants.CM.lower(), {}, {"_id": 0, "featureVector": 1})
    featurearray = np.array(list(map(lambda x: x['featureVector'], list(results))))
    model = Model()
    model.save_model(featurearray, 'cm_np')


def dimension_reduction():
    # save_model_file()
    constants = GlobalConstants()
    model = Model()
    features = model.load_model('cm_np')
    redn = DimensionReduction(dimension_reduction_model=constants.PCA, extractor_model=constants.CM, matrix=features,
                              conversion=True, k_value=500)
    redn.execute()
    pass


def task5a(layers=10, k=10, combine_models=False):
    constants = GlobalConstants()
    xt = time.time()
    model = Model()
    if combine_models:
        data0 = model.load_model(constants.HOG_NMF_MODEL_FILE)
        data1 = model.load_model(constants.CM_PCA_MODEL_FILE)
        data = np.concatenate((data0, data1), axis=1)
    else:
        data = model.load_model(constants.HOG_NMF_MODEL_FILE)

    lsh = LSH(layers=layers, khash_count=k, w=constants.LSH_W, image_ids=img_ids(), data=data)
    l_hashes, l_buckets = lsh.create_index()
    model.save_model(lsh, constants.LSH_OBJECT)
    model.save_model(l_hashes, constants.LSH_L_HASHES)
    model.save_model(l_buckets, constants.LSH_L_BUCKETS)
    print(time.time() - xt)


def task5b(query, top, visualize=False, combine_models=False):
    constants = GlobalConstants()
    lsh = Model().load_model(constants.LSH_OBJECT)
    imageids, feat_vectors, query_vector = lsh.query(query, top)
    print(imageids[:top])
    print("Unique images: {}".format(str(lsh.get_shape()[0] - 1)))
    if visualize:
        result = []
        for rank, image in enumerate(imageids[:top]):
            res = {'path': os.path.join("Hands", image), 'imageId': image, 'rank': rank+1}
            result.append(res)
        if combine_models:
            extract = "HOG + CM"
        else:
            extract = "HOG"
        title = {
            "Search": "Locality Sensitive Hashing (LSH)",
            "Feature Extraction": extract,
            "L": lsh.get_l(),
            "K": lsh.get_k(),
            "Dimensionality Reduction": "NMF",
            "t": 20,
            "Distance": "Euclidean"
        }
        print(os.path.abspath(os.path.join("Hands", query)))
        show_images(os.path.abspath(os.path.join("Hands", query)), result, title, rank=True)

    return imageids, feat_vectors, query_vector


def img_ids():
    constants = GlobalConstants()
    image_names = list(MongoWrapper(constants.Mongo().DB_NAME).find(constants.CM.lower(), {}, {"_id": 0, "imageId": 1}))
    return list(map(lambda x: x['imageId'], image_names))


if __name__ == '__main__':
    # task5a(layers=10, k=10, combine_models=False)
    task5b("Hand_0000674.jpg", 20, False, False)
    # dimension_reduction()
    # save_model_file()
    # img_ids()
