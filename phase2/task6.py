import operator
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import random
import time
import warnings

import numpy as np
import pandas as pd
import phase2.task1 as task1

import utils.imageviewer as imgvwr
from classes.dimensionreduction import DimensionReduction
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper
from utils.inputhelper import get_input_subject_id
from utils.model import Model

warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)

feature_extraction_model = "HOG"
dimension_reduction_model = "NMF"
dist_func = "euclidean"


def find_similar_subjects(given_subject_id, image_list_for_given_subject, given_model, dataset_images, dim_reduction,m_value, folder):
    
    print("\nComputing subject-subject similarity for subject ID:", given_subject_id)

    distance_for_image = []
    for img in (image_list_for_given_subject):
        # calculate distance for each image of subject from all other images in dataset
        distance_for_image.append(dim_reduction.find_m_similar_images(given_model, m_value , folder, img, dist_func))

    # get the metadata for each image with given subject id
    subject_data = dim_reduction.get_metadata("imageName", dataset_images['imageId'].tolist())
    # unique subject IDs in dataset
    dataset_subject_ids = set((subject_data)["id"])
    distance_for_subject = []
    for image_image_distances in distance_for_image:
        for subject in dataset_subject_ids:
            # if subject == given_subject_id:
            #     continue
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
    return subject_similarity


def load_model(dim_reduction, feature_extraction_model, dimension_reduction_model, k_value):
    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model = model_interact.load_model(filename=filename)
    if model is None:
        print("Saving the model")
        task1.save_model(dim_reduction,feature_extraction_model,dimension_reduction_model,k_value)
        model = model_interact.load_model(filename=filename)
    return model


def main():
    # given subject id
    given_subject_id = get_input_subject_id()
    k_value = 40
    master_folder = "Hands"
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    # original feature vectors
    obj_feat_matrix = dim_reduction.get_object_feature_matrix()
    # extract model saved from task 1
    model = load_model(dim_reduction, feature_extraction_model, dimension_reduction_model, k_value)
    # get the img IDs from the database for images in the fit model
    img_set = pd.DataFrame({"imageId": obj_feat_matrix['imageId']})
    # image count to rank against current image
    m_value = len(img_set)
    print(global_constants.LINE_SEPARATOR)
    print("User Inputs summary")
    print(global_constants.LINE_SEPARATOR)
    print("Query Subject Id: {}".format(given_subject_id))
    print(global_constants.LINE_SEPARATOR)
    # given_subject_id = 55
    # similar subjects to find
    similar_subject_count = 3
    # get metadata for given subject's images
    metadata = dim_reduction.get_metadata("id", list([given_subject_id]))
    # get a list of img IDs for the particular subject in the dataset
    image_list_for_given_subject = random.sample(list(set(metadata["imageName"].tolist())), 5)
    image_list = list(set(img_set["imageId"].tolist()))

    starttime = time.time()

    # method call to find similar subjects
    subject_similarity = find_similar_subjects(given_subject_id, image_list_for_given_subject, model,
                                                                    img_set, dim_reduction,m_value, master_folder)
    # sort the similarity scores in descending order 
    sorted_subject_similarity = sorted(subject_similarity.items(), key=operator.itemgetter(1), reverse=True)

    print()
    print("Subject  :   Score")
    list_subjects = []
    max = similar_subject_count
    counter = 0
    while counter < max:
        subject = sorted_subject_similarity[counter]
        if subject[0] != given_subject_id:
            print(subject[0],"  :   ",subject[1])
            list_subjects.append([subject[0],subject[1]])
        else:
            max +=1
        counter+=1
    print()
    # print(sorted_subject_similarity)

    image_list_for_similar_subjects_abs_path = []
    similarity_scores = []
    folder_path = os.path.dirname(obj_feat_matrix['path'][0])
    # create list of images for each subject to visualize most similar subjects
    for subject in (sorted_subject_similarity):
        if subject[0] != given_subject_id:
            metadata = dim_reduction.get_metadata("id", list([subject[0]]))
            similarity_scores.append(subject[1])
            image_list_for_similar_subject = list(set(metadata["imageName"].tolist()).intersection(set(img_set["imageId"].tolist())))
            image_list_for_one_similar_subject_abs_path = []
            for image in image_list_for_similar_subject:
                image_list_for_one_similar_subject_abs_path.append((os.path.join(folder_path,image)))
            image_list_for_similar_subjects_abs_path.append(image_list_for_one_similar_subject_abs_path)
            similar_subject_count -=1
            if (similar_subject_count <= 0):
                break

    # Create image list for given subject
    image_list_for_given_subject_abs_path = []
    # pick 5 images of given subject at random from master dataset
    for image in image_list_for_given_subject:
        image_list_for_given_subject_abs_path.append(os.path.abspath(os.path.join(master_folder,image)))

    output_path = os.path.abspath(os.path.join("output"))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig_filename = os.path.join(output_path,"task6_{0}_{1}_{2}_{3}_{4}.png".format(feature_extraction_model,
                                        dimension_reduction_model, str(k_value),dist_func,given_subject_id))
    # show images on a plot
    imgvwr.show_subjectwise_images(given_subject_id, image_list_for_given_subject_abs_path,
                                         list_subjects, image_list_for_similar_subjects_abs_path, fig_filename)

    print("\nTime taken for task 6: {}\n".format(time.time() - starttime))

if __name__ == "__main__":
    main()
