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
import phase2.task1 as p1task1
import phase2.task5 as p2task5
import phase2.task6 as p2task6
import time
import random
import warnings
warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)

feature_extraction_model = "HOG"
dimension_reduction_model = "NMF"
dist_func = "euclidean"

def compute_latent_semantic(label, k_value):
    
    pass

def main():


    """
            folder = get_input_folder()
            model = get_input_feature_extractor_model()

            feature_extractor = ExtractFeatures(folder, model)
            feature_extractor.execute()


    """
    k_value = 10
    # query_folder = get_input_folder()
    training_set = 'C:\mwdb\commoncode\CSE515\Dataset3\Labelled\Set1'
    test_set = 'C:\mwdb\commoncode\CSE515\Dataset3\\Unlabelled\Set 1'

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)

    dim_reduction.execute()
    p2task5.run_task3(feature_extraction_model, dimension_reduction_model, "dorsal", k_value)

    result_dorsal = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_dorsal.append((p2task5.run_task4(feature_extraction_model, dimension_reduction_model, 
                            training_set, image_name, dist_func, "dorsal", k_value, m_value=1))[0]['score'])
    p2task5.run_task3(feature_extraction_model, dimension_reduction_model, "palmar", k_value)
    result_palmar = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_palmar.append((p2task5.run_task4(feature_extraction_model, dimension_reduction_model, 
                            training_set, image_name, dist_func, "palmar", k_value, m_value=1))[0]['score'])

    for root, dirs, images in os.walk(test_set):
        for i, image_name in enumerate(images):
            if result_dorsal[i] > result_palmar[i]:
                print(image_name, ' : dorsal')
            else:
                print(image_name, ' : palmar')

if __name__ == "__main__":
    main()
