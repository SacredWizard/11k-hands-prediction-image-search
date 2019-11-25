import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
from classes.mongo import MongoWrapper
from classes.globalconstants import GlobalConstants
from utils.inputhelper import get_input_subject_id
import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
import utils.imageviewer as imgvwr
from utils.excelcsv import CSVReader
import phase2.task1 as p1task1
import phase2.task5 as p2task5
import phase2.task6 as p2task6
import time
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)
csv_reader = CSVReader()

def compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label, k_value, folder):

    # p2task5.run_task3(feature_extraction_model, dimension_reduction_model, label, k_value)

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, label, folder_metadata=folder)
    obj_lat, feat_lat, model = dim_reduction.execute()
    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label,
                                        str(k_value))
    model_interact.save_model(model=model, filename=filename)
    return obj_lat,feat_lat, model


def main():
    """
    folder = get_input_folder()
    model = get_input_feature_extractor_model()

    feature_extractor = ExtractFeatures(folder, model)
    feature_extractor.execute()
    """
    feature_extraction_model = "HOG"
    dimension_reduction_model = "NMF"
    dist_func = "euclidean"
    k_value = 12
    # query_folder = get_input_folder()
    training_set = 'C:\mwdb\commoncode\CSE515\Dataset3\Labelled\Set2'
    test_set = 'C:\mwdb\commoncode\CSE515\Dataset3\\Unlabelled\Set 1'
    label = "dorsal"
    obj_lat_d,feat_lat_d, model_d = compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label, str(k_value))
    csv_reader.save_to_csv(obj_lat_d, feat_lat_d, filename)

    label = "palmar"
    obj_lat_p,feat_lat_p, model_p = compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(feature_extraction_model, dimension_reduction_model, label, str(k_value))
    csv_reader.save_to_csv(obj_lat_p, feat_lat_p, filename)

    # x_train, x_test, y_train, y_test = feat_lat_d, test_set, feat_lat_p, test_set
    # lr = LogisticRegression()
    # lr.fit(x_train, y_train)
    # x_pred = lr.predict(x_test)
    # y_pred = lr.predict(y_test)

    result_dorsal = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_dorsal.append((p2task5.run_task4(feature_extraction_model, dimension_reduction_model, 
                            test_set, image_name, dist_func, "dorsal", k_value, m_value=1))[0]['score'])
    result_palmar = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_palmar.append((p2task5.run_task4(feature_extraction_model, dimension_reduction_model, 
                            test_set, image_name, dist_func, "palmar", k_value, m_value=1))[0]['score'])

    for root, dirs, images in os.walk(test_set):
        for i, image_name in enumerate(images):
            if result_dorsal[i] > result_palmar[i]:
                print(image_name, ' : dorsal')
            else:
                print(image_name, ' : palmar')


if __name__ == "__main__":
    main()

    