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
import task1 as task1
import time

model_interact = Model()
mongo_wrapper = MongoWrapper(GlobalConstants().Mongo().DB_NAME)
feature_extraction_model = "CM"
dimension_reduction_model = "SVD"
k_value = 40
dist_func = "manhattan"
dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
folder = ""
# original feature vectors
obj_feat_matrix = dim_reduction.get_object_feature_matrix()
# extract model saved from task 1
filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
model = model_interact.load_model(filename=filename)
if model is None:
    task1.save_model(dim_reduction,feature_extraction_model,dimension_reduction_model,k_value)
    model = model_interact.load_model(filename=filename)
# get the img IDs from the database for images in the fit model
img_set = pd.DataFrame({"imageId": obj_feat_matrix['imageId']})
# image count to rank against current image
m_value = len(img_set)


def find_similar_subjects(given_subject_id):
    
    # get metadata for all images
    metadata = dim_reduction.get_metadata("id", list([given_subject_id]))
    # get a list of img IDs for the particular subject in the dataset
    image_list_for_given_subject = list(set(metadata["imageName"].tolist()).intersection(set(img_set["imageId"].tolist())))
    distance_for_image = []
    for img in (image_list_for_given_subject):
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
            list_image_subject_scores = list( (float(d['score']) for d in image_image_distances if int(d['subject']) == subject))
            # take max similarity score of images of each subject as image-subject score 
            max_image_subject_scores = max(list_image_subject_scores)
            # get all the similarity scores for all images of given subject with each subject in database
            distance_for_subject.append({"subject":subject , "score":max_image_subject_scores})
    subject_similarity = {}
    for subject in dataset_subject_ids:
        # get the average subject-subject similarity scores
        subject_similarity[subject] = np.mean(list(float(d['score']) for d in distance_for_subject if d['subject'] == subject ))
    # sort the similarity scores in descending order 
    return subject_similarity


def main():
    # given subject id
    given_subject_id = get_input_subject_id()
    # similar subjects to find
    similar_subject_count = 3
    starttime = time.time()

    # method call to find similar subjects
    subject_similarity = find_similar_subjects(given_subject_id)
    sorted_subject_similarity = sorted(subject_similarity.items(), key=operator.itemgetter(1), reverse=True)

    print()
    print("Subject  :   Score")
    list_subjects = []
    loop_break = 0
    for subject in sorted_subject_similarity:
        if not int(subject[0]) == given_subject_id:
            print(subject[0],"  :   ",subject[1])
            loop_break+=1
        list_subjects.append([subject[0],subject[1]])
        if loop_break >= similar_subject_count:
            break
    print()
    # print(sorted_subject_similarity)

    image_list_for_similar_subjects_abs_path = []
    similarity_scores = []
    path = os.path.dirname(obj_feat_matrix['path'][0])
    # create list of images for each subject to visualize most similar subjects
    for subject in (sorted_subject_similarity):
        metadata = dim_reduction.get_metadata("id", list([subject[0]]))
        similarity_scores.append(subject[1])
        image_list_for_similar_subject = list(set(metadata["imageName"].tolist()).intersection(set(img_set["imageId"].tolist())))
        image_list_for_one_similar_subject_abs_path = []
        for image in image_list_for_similar_subject:
            image_list_for_one_similar_subject_abs_path.append((os.path.join(path,image)))

        image_list_for_similar_subjects_abs_path.append(image_list_for_one_similar_subject_abs_path)
        similar_subject_count -=1
        if (similar_subject_count < 0):
            break
    # show images on a plot
    imgvwr.show_subjectwise_images(list_subjects, image_list_for_similar_subjects_abs_path)

    print("\nTime taken for task 6: {}\n".format(time.time() - starttime))

if __name__ == "__main__":
    main()
