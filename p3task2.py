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
import phase3.task1 as p3task1

from itertools import groupby
import time
import random
import warnings

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)

feature_extraction_model = "LBP"
dimension_reduction_model = "LDA"
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

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)

    # obj_lat, feat_lat, model = dim_reduction.execute()
    training_set = ('/Users/kanishkbashyam/Desktop/CSE515/Dataset3/Labelled/Set2')
    label = 'dorsal'
    obj_lat, feat_lat, model = p3task1.compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label, k_value, training_set)
    label_p = 'palmar'
    obj_lat_p,feat_lat_p, model_p = p3task1.compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label_p, k_value, training_set)
    test_set = ('/Users/kanishkbashyam/Desktop/CSE515/Dataset3/Unlabelled/Set2')
    red_dim = p3task1.reduced_dimensions_for_unlabelled_folder(feature_extraction_model, dimension_reduction_model, k_value, label, training_set, test_set)


   
    df = obj_lat[['reducedDimensions','imageId']]
    
    tf = obj_lat['reducedDimensions']
    tf_p = obj_lat_p['reducedDimensions']

    a=[]
    a_p=[]
    for x in tf:
        a.append(x)
    for x in tf_p:
        a_p.append(x)

    X= df.values
    
    k=5
    #    
    km = KMeans(n_clusters=k, random_state=0).fit(a)
    km_p = KMeans(n_clusters=k, random_state=0).fit(a_p)


    # print(km.labels_)
    counter = np.zeros(5)
    for k_m in km.labels_:
        counter[k_m] +=1
    print(counter)
    # 
    d_cluster = km.predict(red_dim['reducedDimensions'].tolist())
    p_cluster = km_p.predict(red_dim['reducedDimensions'].tolist())

    unlabelled_aspect = dim_reduction.get_metadata("imageName", red_dim['imageId'].tolist())['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]

    good=0
    bad=0
    for ind in range(len(red_dim['reducedDimensions'])):
        # if km.cluster_centers_
        cc_dorsal = km.cluster_centers_[d_cluster[ind]]
        cc_palmar = km_p.cluster_centers_[p_cluster[ind]]
        dist_dorsal = np.linalg.norm(red_dim['reducedDimensions'][ind]-cc_dorsal)
        dist_palmar = np.linalg.norm(red_dim['reducedDimensions'][ind]-cc_palmar)
        if dist_dorsal<dist_palmar:
            print(red_dim['imageId'][ind], label, y_test[ind])
            if y_test[ind] == label:
                good +=1
            else:
                bad+=1
        else:
            print(red_dim['imageId'][ind], 'palmar', y_test[ind])
            if y_test[ind] == label_p:
                good +=1
            else:
                bad+=1
        
    print ("good",good)        
    print("bad",bad)
    # km.score()
    random=np.random.choice(len(X),size=k,replace=False)
    print(random)
    centroid={}
    classes={}

    # for i in range(k):
    #     centroid[i]= X[random[i]][0]

    # for iter in range(500):
        
    #     classes2={}
    #     for i in range (k):
    #         classes[i]=[]
    #         classes2[i]=[]
    #         distance=[]

    
    #     for x in X:
    #         # print(x[1])
    #         distance=[np.linalg.norm(np.asarray(x[0]) - np.asarray(centroid[ind])) for ind in range(len(centroid)) ]
        
     
    #         classification = distance.index(min(distance))
    #         classes[classification].append(x)
    #         classes2[classification].append(x[0])
    #     previous = dict(centroid)

    #     for classification in classes2:
    #         centroid[classification]=np.average(classes2[classification], axis=0)

    #     opti = 0

    #     for c in centroid:
            
    #         og_c = previous[c]
    #         current = centroid [c]
    #         if (np.array_equal(current,og_c)):
    #             opti += 1

    #     if (opti==(k)):
    #         # print(iter)
    #         break

            
        # print(iter)            

        # if isOptimal:
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #     print(j)
        #     break

    
    # print("................")
    # for j in range(k):
    #     # for i in range(len(classes[j])):
    #     print((len(classes[j])))
    #     print("??????????????????????")


    


    

            
        

if __name__ == "__main__":
    main()
