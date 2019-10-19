from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
from classes.mongo import MongoWrapper
from classes.globalconstants import GlobalConstants
import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
import task6 as task6
import os.path as path
from utils.inputhelper import get_input_k
import time
from utils.termweight import print_tw
model_interact = Model()
mongo_wrapper = MongoWrapper(GlobalConstants().Mongo().DB_NAME)


def main():
    feature_extraction_model = "CM"
    dimension_reduction_model = "SVD"
    k_value = 40

    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    # original feature vectors
    obj_feat_matrix = dim_reduction.get_object_feature_matrix()
        # get the img IDs from the database for images in the fit model
    img_set = pd.DataFrame({"imageId": obj_feat_matrix['imageId']})
    # get the metadata for each image with given subject id
    subject_data = dim_reduction.get_metadata("imageName", list(set(img_set["imageId"].tolist())))
    # unique subject IDs in dataset
    dataset_subject_ids = set((subject_data)["id"])
    subject_subject_matrix = []

    starttime = time.time()
    for i,subjectid in enumerate(dataset_subject_ids):
        print("Computing subject similarity for subject ID:",subjectid)
        similar_subjects = task6.find_similar_subjects(subjectid)
        subject_subject_matrix.append(np.asarray(list(similar_subjects.values())))

    print("\nTime taken to create subject subject matrix: {}\n".format(time.time() - starttime))
    # perform nmf on subject_subject_matrix
    given_k_value = get_input_k()
    # given_k_value = 1
    matrix = pd.DataFrame(data = {'imageId':list(dataset_subject_ids),'featureVector': subject_subject_matrix})
    dim_red = DimensionReduction(None, "NMF", given_k_value, subject_subject=True, matrix=matrix)
    w, h, model = dim_red.execute()

    # display latent semantics
    # printing the term weight
    print_tw(w, h, subject_subject=True)
    # save to csv
    filename = "task7" + '_' + feature_extraction_model + '_' + dimension_reduction_model + '_' + str(given_k_value)
    CSVReader().save_to_csv(w, None, filename, subject_subject=True)

    print("\nTime taken for task 7: {}\n".format(time.time() - starttime))

if __name__ == "__main__":
    main()
