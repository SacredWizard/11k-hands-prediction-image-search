
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
model_interact = Model()
mongo_wrapper = MongoWrapper(GlobalConstants().Mongo().DB_NAME)


def find_similar_subjects(given_subject_id):
    
    feature_extraction_model = "HOG"
    dimension_reduction_model = "SVD"
    dist_func = "euclidean"
    folder = ""
    k_value = 10
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    # get metadata for all images
    metadata = dim_reduction.get_metadata("id", list([given_subject_id]))
    # original feature vectors
    obj_feat_matrix = dim_reduction.get_object_feature_matrix()
    # extract model saved from task 1
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model = model_interact.load_model(filename=filename)
    # get the img IDs from the database for images in the fit model
    img_set = pd.DataFrame({"imageId": obj_feat_matrix['imageId']})
    # image count to rank against current image
    m_value = len(img_set)
    # get a list of img IDs for the particular subject in the dataset
    image_list = list(set(metadata["imageName"].tolist()).intersection(set(img_set["imageId"].tolist())))
    distance_for_image = []
    for img in (image_list):
        # calculate distance for each image of subject from all other images in dataset
        distance_for_image.append(dim_reduction.find_m_similar_images(model, m_value , folder, img, dist_func))

    # get the metadata for each image with given subject id
    subject_data = dim_reduction.get_metadata("imageName", list(set(img_set["imageId"].tolist())))
    # unique subject IDs in dataset
    dataset_subject_ids = set((subject_data)["id"])
    
    distance_for_subject = []
    for image_image_distances in distance_for_image:
        for subject in dataset_subject_ids:
            # get the list of similarity scores for each subject against given image
            list_image_subject_scores = (float(d['score']) for d in image_image_distances if int(d['subject']) == subject)
            # take an average of subject similarity score to each image of given subject 
            mean_image_subject_scores = np.mean(list(list_image_subject_scores))
            # get all the similarity scores for all images of given subject with each subject in database
            distance_for_subject.append({"subject":subject , "score":mean_image_subject_scores})
    subject_similarity = {}
    for subject in dataset_subject_ids:
        # get the average subject-subject similarity scores
        subject_similarity[subject] = np.mean(list(float(d['score']) for d in distance_for_subject if d['subject'] == subject ))
    # sort the similarity scores in descending order 
    sorted_subject_similarity = sorted(subject_similarity.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_subject_similarity


def main():
    # given subject id
    given_subject_id = get_input_subject_id()
    # similar subjects to find
    similar_subject_count = 3
    # method call to find similar subjects
    similar_subjects = find_similar_subjects(given_subject_id)
    print()
    print("Subject  :   Score")
    loop_break = 0
    for subject in similar_subjects:
        if int(subject[0]) == given_subject_id:
            continue
        else:
            print(subject[0],"  :   ",subject[1])
            loop_break+=1
        if loop_break >= similar_subject_count:
            break
    print()
    # print(sorted_subject_similarity)


if __name__ == "__main__":
    main()
