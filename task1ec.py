"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for task 1 of Phase 2 of the project
"""
from classes.dimensionreduction import DimensionReduction
from utils.termweight import print_tw
from utils.model import Model
from utils.excelcsv import CSVReader
import pandas as pd
import numpy as np
from utils.imageviewerec import show_images
model_interact = Model()


def main():
    """Main function for the task 1"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "PCA"
    k_value = 10

    # Performs the dimensionality reduction
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    obj_lat, feat_lat, model = dim_reduction.execute()

    # Saves the returned model
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model_interact.save_model(model=model, filename=filename)

    Print term weight pairs to terminal  
    data_tw, feature_tw = print_tw(obj_lat, feat_lat)

    # save term weight pairs to csv  
    filename = "task1"+'_'+feature_extraction_model+'_'+dimension_reduction_model+'_'+str(k_value)
    CSVReader().save_to_csv(obj_lat, feat_lat, filename)
    
    data = dim_reduction.get_object_feature_matrix()
    
    #extra credit for feature latent semantic
    images_list = []
    for f in feat_lat:
        k_list = [f.dot(data['featureVector'][i]) for i in range(len(data['featureVector']))]
        index = np.argsort(-np.array(k_list))[0]
        print(index)
        rec = dict()
        rec['imageId'] = data['imageId'][index]
        rec['path'] = data['path'][index]
        images_list.append(rec)
    
    show_images(images_list)

if __name__ == "__main__":
    main()
