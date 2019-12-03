import time

import numpy as np

from classes.LSH import LSH
from classes.dimensionreduction import DimensionReduction
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper
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


def task5a(layers=10, k=10):
    constants = GlobalConstants()
    xt = time.time()
    model = Model()
    data = model.load_model('lsh_nmf_w')
    lsh = LSH(layers=layers, khash_count=k, w=constants.LSH_W, image_ids=img_ids(), data=data)
    l_hashes, l_buckets = lsh.create_index()
    model.save_model(lsh, constants.LSH_OBJECT)
    model.save_model(l_hashes, constants.LSH_L_HASHES)
    model.save_model(l_buckets, constants.LSH_L_BUCKETS)
    print(time.time() - xt)


def task5b(query, top):
    constants = GlobalConstants()
    lsh = Model().load_model(constants.LSH_OBJECT)
    imageids, feat_vectors = lsh.query(query, top)
    print(imageids[:top])
    print("Overall images: {}".format(len(imageids)))


def img_ids():
    constants = GlobalConstants()
    image_names = list(MongoWrapper(constants.Mongo().DB_NAME).find(constants.CM.lower(), {}, {"_id": 0, "imageId": 1}))
    return list(map(lambda x: x['imageId'], image_names))


if __name__ == '__main__':
    # task5a()
    task5b("Hand_0000003.jpg", 20)
    # dimension_reduction()
    # save_model_file()
    # img_ids()
