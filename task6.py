
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
from classes.mongo import MongoWrapper
from classes.globalconstants import GlobalConstants
import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
model_interact = Model()
mongo_wrapper = MongoWrapper(GlobalConstants().Mongo().DB_NAME)
def main():
    # given_subject_id
    given_subject_id = 27
    feature_extraction_model = "HOG"
    dimension_reduction_model = "SVD"
    dist_func = "euclidean"
    m_value = 200
    folder = ""
    k_value = 10
    dim_reduction = DimensionReduction(feature_extraction_model, dimension_reduction_model, k_value)
    # get metadata for all images
    metadata = dim_reduction.get_metadata("id", list([given_subject_id]))

    obj_feat_matrix = dim_reduction.get_object_feature_matrix()

    filename = feature_extraction_model + "_" + dimension_reduction_model + "_" + str(k_value)
    model = model_interact.load_model(filename=filename)
    # get the img IDs from the database for images in the fit model
    img_set = pd.DataFrame({"imageId": obj_feat_matrix['imageId']})
    # get an intersection of img IDs for images for the particular subject
    image_list = list(set(metadata["imageName"].tolist()).intersection(set(img_set["imageId"].tolist())))
    distance_for_image = []
    for img in (image_list):
        distance_for_image.append(dim_reduction.find_m_similar_images(model, m_value , folder, img, dist_func))

    subject_data = dim_reduction.get_metadata("imageName", list(set(img_set["imageId"].tolist())))
    dataset_subject_ids = set((subject_data)["id"])
    
    distance_for_subject = []
    for i,image_image_distances in enumerate(distance_for_image):
        for subject in dataset_subject_ids:
            list_image_subject_scores = (float(d['score']) for d in image_image_distances if int(d['subject']) == subject)
            mean_image_subject_scores = np.mean(list(list_image_subject_scores))
            distance_for_subject.append({"subject":subject , "score":mean_image_subject_scores})
        # distance_for_subject.append(row)
    subject_similarity = {}
    for subject in dataset_subject_ids:
        subject_similarity[subject] = np.mean(list(float(d['score']) for d in distance_for_subject if d['subject'] == subject ))

    sorted_subject_similarity = sorted(subject_similarity.items(), key=operator.itemgetter(1), reverse=True)

    print(subject_similarity)
    print()
    print(sorted_subject_similarity)
    # (obj_feature.loc[obj_feature['imageId'] == image])["featureVector"]

if __name__ == "__main__":
    main()
