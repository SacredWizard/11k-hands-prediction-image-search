import os
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
dimension_reduction_model = "PCA"
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
    k_value = 30
    
    

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value, "dorsal")

    obj_lat, feat_lat, model = dim_reduction.execute()

    
    
   
    df = obj_lat[['reducedDimensions','imageId']]
    
    

    
    X= df.values
    
    
   

    

    

    k=5

    random=np.random.randint(0,len(X),size=5)
    centroid={}
    classes={}

    for i in range(k):
        centroid[i]= X[random[i]][0]

    for j in range(500):
        
        classes2={}
        for i in range (k):
            classes[i]=[]
            classes2[i]=[]
            distance=[]

    
        for x in X:
            # print(x[1])
            distance=[np.linalg.norm(np.asarray(x[0]) - np.asarray(centroid[k])) for k in range(len(centroid)) ]
        
     
            classification = distance.index(min(distance))
            classes[classification].append(x)
            classes2[classification].append(x[0])

            
            

        previous = dict(centroid)

        for classification in classes2:



            centroid[classification]=np.average(classes2[classification], axis=0)

        

        isOptimal = False

        for c in centroid:
            
            og_c= (previous[c])
            current= centroid[c]



            if np.sum((current - og_c)/og_c * 100.0) > 0.0001:
                isOptimal = True
            
       
            

        if isOptimal:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(j)
            break

    
    
    for j in range(k):

    
        for i in range(len(classes[j])):
            print(classes[j][i][1])
        
        print("??????????????????????")

if __name__ == "__main__":
    main()
