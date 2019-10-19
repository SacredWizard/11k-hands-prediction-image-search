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
    for subjectid in dataset_subject_ids:
        subject_subject_matrix.append(list((task6.find_similar_subjects(subjectid)).values()))

    # TODO perform nmf on subject_subject_matrix
    # TODO display latent semantics
    print()
    for subject in subject_subject_matrix:
        print(subject)
        print()

if __name__ == "__main__":
    main()
