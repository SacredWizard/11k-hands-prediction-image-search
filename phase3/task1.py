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
import os
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
    k_value = 40
    
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    p2task5.run_task3(feature_extraction_model, dimension_reduction_model, "dorsal", k_value)

    pass

if __name__ == "__main__":
    pass
