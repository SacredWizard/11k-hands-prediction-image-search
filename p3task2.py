import os
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
from classes.mongo import MongoWrapper
from classes.globalconstants import GlobalConstants
from utils.inputhelper import get_input_folder,get_input_k
import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
import utils.imageviewer as imgvwr
import phase2.task1 as p1task1
import phase2.task5 as p2task5
import phase2.task6 as p2task6
import phase3.task1 as p3task1

from flask import Flask, request, render_template, send_from_directory

from itertools import groupby
import time
import random
import warnings


from sklearn.cluster import KMeans
port_g = 4550
app = Flask(__name__)
warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)

feature_extraction_model = "HOG"
dimension_reduction_model = "SVD"
dist_func = "euclidean"

def compute_latent_semantic(label, k_value):
    
    pass

def main():

    k = get_input_k("C")
    training_set = get_input_folder("Labelled")
    test_set = get_input_folder("Classify")
    k_value = 30

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)

    # obj_lat, feat_lat, model = dim_reduction.execute()
    label = 'dorsal'
    obj_lat, feat_lat, model = p3task1.compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label, k_value, training_set)
    label_p = 'palmar'
    obj_lat_p,feat_lat_p, model_p = p3task1.compute_latent_semantic_for_label(feature_extraction_model, dimension_reduction_model, label_p, k_value, training_set)
    red_dim = p3task1.reduced_dimensions_for_unlabelled_folder(feature_extraction_model, dimension_reduction_model, k_value, label, training_set, test_set)


    #input for project
    df = obj_lat[['reducedDimensions','imageId']]
    df_p = obj_lat_p[['reducedDimensions','imageId']]
    #inputt for scikit
    tf = obj_lat['reducedDimensions']
    tf_p = obj_lat_p['reducedDimensions']

    a=[]
    a_p=[]
    for x in tf:
        a.append(x)
    for x in tf_p:
        a_p.append(x)

    X= df.values
    Y= df_p.values
    

    # k clusters
    # k=5
    #    
    km = KMeans(n_clusters=k, random_state=0,n_init=30,init='k-means++',precompute_distances=True,n_jobs= -1).fit(a)
    km_p = KMeans(n_clusters=k, random_state=0,n_init=30,init='k-means++',precompute_distances=True,n_jobs= -1).fit(a_p)



    # print(km.labels_)
    counter = np.zeros(k)
    counter_p = np.zeros(k)
    for k_m in km.labels_:
        counter[k_m] +=1
    # print(counter)
    for k_m_p in km_p.labels_:
        counter_p[k_m_p] +=1
    # print(counter_p)
    # 
    d_cluster = km.predict(red_dim['reducedDimensions'].tolist())
    p_cluster = km_p.predict(red_dim['reducedDimensions'].tolist())

    unlabelled_aspect = dim_reduction.get_metadata_collection("imageName", red_dim['imageId'].tolist(), "unlabelled")['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]

#min max test
    
    good=0
    bad=0
    # for ind in range(len(red_dim['reducedDimensions'])):
    
    #     cc_dorsal = km.cluster_centers_[d_cluster[ind]]
    #     cc_palmar = km_p.cluster_centers_[p_cluster[ind]]
    #     dist_dorsal = np.linalg.norm(red_dim['reducedDimensions'][ind]-cc_dorsal)
    #     dist_palmar = np.linalg.norm(red_dim['reducedDimensions'][ind]-cc_palmar)
        
  
    #     if dist_dorsal<dist_palmar:
    #         #print(red_dim['imageId'][ind], label, y_test[ind])
    #         if y_test[ind] == label:
    #             good +=1
    #         else:
    #             bad+=1
    #     else:
    #         #print(red_dim['imageId'][ind], 'palmar', y_test[ind])
    #         if y_test[ind] == label_p:
    #             good +=1
    #         else:
    #             bad+=1
        
    # print ("good",good)        
    # print("bad",bad)
    # km.score()
    
    
    def kmeans_implementation(X):
        random=np.random.choice(len(X),size=k,replace=False)
        
        centroid={}
        classes={}
        classes2={}

        
        # for cen in range(k):
        #     for im in range(0,len(X)):
        #         distance=[np.linalg.norm(np.asarray(X[im][0]) - np.asarray(centroid[0])))]

        for i in range(k):
            centroid[i]=X[random[i]][0]

        for iter in range(500):
            
            
            for i in range (k):
                classes[i]=[]
                classes2[i]=[]
                distance=[]

        
            for x in X:
                # print(x[1])
                distance=[np.linalg.norm(np.asarray(x[0]) - np.asarray(centroid[ind])) for ind in range(len(centroid)) ]
            
        
                classification = distance.index(min(distance))
                classes[classification].append(x)
                classes2[classification].append(x[0])
            previous = dict(centroid)

            for classification in classes2:
                centroid[classification]=np.average(classes2[classification], axis=0)

            opti = 0

            for c in centroid:
                
                og_c = previous[c]
                current = centroid [c]
                if (np.array_equal(current,og_c)):
                    opti += 1

            if (opti==(k)):
                # print(iter)
                break

                
                       

        

        
        return classes,centroid



    classes,centroid=kmeans_implementation(X)
    classes_p,centroid_p=kmeans_implementation(Y)
        

    
   
#predict loop red_dimension is the query folder

    def predict_class(red_dim,centroid):
        query_classes={}
        for i in range (k):
                query_classes[i]=[]
                

        for ind in range(len(red_dim['reducedDimensions'])):
            cluster_distance=[]
            cluster_distance=[np.linalg.norm(red_dim['reducedDimensions'][ind] - np.asarray(centroid[q])) for q in range(len(centroid)) ]
            query_classification = cluster_distance.index(min(cluster_distance))
            query_classes[query_classification].append(red_dim['imageId'][ind])
        return query_classes





    query_classes_dorsal  =  predict_class(red_dim,centroid)
    query_classes_palmar  =  predict_class(red_dim,centroid)
    
    
    
    
    
    correct=0
    wrong=0

    def centroid_mean(centroid):
        res_list=[0]*k_value
        mean_centroid=[]
        for i in range(k):

            res_list = [a+b for a,b in zip(res_list, centroid[i])]
        
        for x in res_list:
            mean_centroid.append(x/k)

        return mean_centroid

    mean_centroid_dorsal = centroid_mean(centroid)
    mean_centroid_palmar = centroid_mean(centroid_p)

    dorsal_images=[]
    palmar_images=[]
    for ind in range(len(red_dim['reducedDimensions'])):
        image_center_dorsal=0
        image_center_palmar=0
        image_name = red_dim['imageId'][ind]


        for i in range(k):
            if (image_name in query_classes_dorsal[i]):
                image_center_dorsal = i
            if ( image_name in query_classes_palmar[i]):
                image_center_palmar = i

        
        dorsal_distance = np.linalg.norm(red_dim['reducedDimensions'][ind]- centroid[image_center_dorsal])
        palmar_distance = np.linalg.norm(red_dim['reducedDimensions'][ind]-  centroid_p[image_center_palmar])

        if dorsal_distance<palmar_distance:
            #print(red_dim['imageId'][ind], label, y_test[ind])´
            dorsal_images.append(red_dim['imageId'][ind])
            if y_test[ind] == label:

                correct+=1
            else:
                wrong +=1
        else:

            #print(red_dim['imageId'][ind], 'palmar', y_test[ind])
            palmar_images.append(red_dim['imageId'][ind])
            if y_test[ind] == label_p:
                correct +=1
            else:
                wrong+=1



    print("correct"+str(correct))
    print("wrong"+str(wrong))
    
    
    
    
    
    
    print("\nClick here: http://localhost:{0}/result\n".format(port_g))
    print("\nClick here: http://localhost:{0}/dorsal\n".format(port_g))
    print("\nClick here: http://localhost:{0}/palmar\n".format(port_g))

    



    
    

    
    


    # APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    

    @app.route('/Dataset2/<filename>')
    def send_image(filename):
        return send_from_directory((training_set), filename)

    @app.route('/test_set/<filename>')
    def send_image_result(filename):
        return send_from_directory((test_set),filename)

    @app.route('/dorsal')
    def get_gallery():
        image_names=[classes,k]
       
        
        return render_template("demo.html", image_names=image_names)

    @app.route('/palmar')
    def get_gallery_p():
        image_names_p=[classes_p,k]
       
        
        return render_template("demo_p.html", image_names_p=image_names_p)


    @app.route('/result')
    def get_gallery_result():
        results = [dorsal_images,palmar_images]

       
        
        return render_template("task2.html", results=results)

    app.run(port=port_g)

if __name__ == "__main__":

    main()
