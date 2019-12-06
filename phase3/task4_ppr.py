"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is the CLI for loading the metadata on to mongo
"""
from classes.dimensionreduction import DimensionReduction
from sklearn.metrics.pairwise import cosine_similarity
from utils.inputhelper import get_input_folder
import time
from classes.mongo import MongoWrapper
import numpy as np


def ppr(sim_graph, images_list, query_images, max_iter=500, alpha=0.85):
    sim_graph = sim_graph.T
    teleport_matrix = np.array([0 if img not in query_images else 1 for img in images_list]).reshape(len(images_list),
                                                                                                     1)
    teleport_matrix = teleport_matrix / len(query_images)
    uq_new = teleport_matrix
    uq_old = np.array((len(images_list), 1))
    iter = 0
    while iter < max_iter and not np.array_equal(uq_new, uq_old):
        uq_old = uq_new.copy()
        uq_new = alpha * np.matmul(sim_graph, uq_old) + (1 - alpha) * teleport_matrix
        iter += 1
    print("Iterations: {}".format(iter))
    uq_new = uq_new.ravel()
    # uq_new = uq_new[::-1].argsort(axis=0)
    a = (-uq_new).argsort()
    result = []
    rank = 1
    res_dict = {}
    for i in a:
        res = {"imageId": images_list[i], "score": uq_new[i], "rank": rank}
        res_dict[images_list[i]] = rank
        result.append(res)
        # print("Image: {} Score: {} Rank:{}".format(images_list[i], uq_new[i], rank))
        rank += 1
    # return result
    return res_dict


def filter_images_by_label(images_list):
    mongo_wrap = MongoWrapper()
    results = mongo_wrap.find("labelled", {"imageName": {"$in": list(images_list)}}, {"_id": 0, "aspectOfHand": 1,
                                                                                "imageName": 1})
    dorsal_img_list = []
    palmar_img_list = []
    for r in results:
        if "dorsal" in r['aspectOfHand']:
            dorsal_img_list.append(r['imageName'])
        elif "palmar" in r['aspectOfHand']:
            palmar_img_list.append(r['imageName'])
    return dorsal_img_list, palmar_img_list


def fetch_actual_labels(images_list):
    mongo_wrap = MongoWrapper()
    results = mongo_wrap.find("metadata", {"imageName": {"$in": list(images_list)}}, {"_id": 0, "aspectOfHand": 1,
                                                                                "imageName": 1})
    final_result = {}
    for r in results:
        if "dorsal" in r["aspectOfHand"]:
            final_result[r["imageName"]] = "dorsal"
        elif "palmar" in r["aspectOfHand"]:
            final_result[r["imageName"]] = "palmar"
    return final_result


def main():
    """Main function for the script"""
    feature_extraction_model = "HOG"
    # feature_extraction_models = ["CM", "HOG"]
    feature_extraction_model_1 = "CM"
    dimension_reduction_model = "PCA"
    k_value = 10
    dim_k_value = 40
    # K_value = 20
    # lab_folder = "Dataset3/Labelled/Set1"
    # unlab_folder = "Dataset3/Unlabelled/Set 2"
    lab_folder = get_input_folder("Labelled Folder")
    unlab_folder = get_input_folder("Classify")
    start = time.time()
    # ================================================================================================================
    # labelled Images
    dim_red = DimensionReduction(feature_extraction_model, dimension_reduction_model, dim_k_value,
                                 folder_metadata=lab_folder,
                                 metadata_collection="labelled")
    obj_feat_lab = dim_red.get_object_feature_matrix()
    features_list_lab = np.array(obj_feat_lab['featureVector'].tolist())
    images_list_lab = np.array(obj_feat_lab['imageId'])
    # filtering the labelled set
    dorsal_list, palmar_list = filter_images_by_label(images_list_lab)

    # unlabelled images
    dim_red = DimensionReduction(feature_extraction_model, dimension_reduction_model, dim_k_value,
                                 folder_metadata=unlab_folder,
                                 metadata_collection="unlabelled")
    obj_feat_unlab = dim_red.get_object_feature_matrix()
    features_list_unlab = np.array(obj_feat_unlab['featureVector'].tolist())
    images_list_unlab = np.array(obj_feat_unlab['imageId'])

    # ================================================================================================================
    # labelled Images
    dim_red = DimensionReduction(feature_extraction_model_1, dimension_reduction_model, dim_k_value,
                                 folder_metadata=lab_folder,
                                 metadata_collection="labelled")
    obj_feat_lab_1 = dim_red.get_object_feature_matrix()
    features_list_lab_1 = np.array(obj_feat_lab_1['featureVector'].tolist())
    # images_list_lab = np.array(obj_feat_lab_1['imageId'])
    # filtering the labelled set


    # unlabelled images
    dim_red = DimensionReduction(feature_extraction_model_1, dimension_reduction_model, dim_k_value,
                                 folder_metadata=unlab_folder,
                                 metadata_collection="unlabelled")
    obj_feat_unlab_1 = dim_red.get_object_feature_matrix()
    features_list_unlab_1 = np.array(obj_feat_unlab_1['featureVector'].tolist())
    # images_list_unlab = np.array(obj_feat_unlab['imageId'])
    features_list_lab = np.concatenate((features_list_lab, features_list_lab_1), axis=1)
    features_list_unlab = np.concatenate((features_list_unlab, features_list_unlab_1), axis=1)

    # ================================================================================================================

    dorsal_list, palmar_list = filter_images_by_label(images_list_lab)
    features_list = np.concatenate((features_list_lab, features_list_unlab))
    images_list = np.concatenate((images_list_lab, images_list_unlab))
    images_list = list(images_list)
    # Finding Similarity Matrix
    cos_sim = cosine_similarity(features_list)
    sim_graph = np.empty((0, len(cos_sim)))
    for row in cos_sim:
        k_largest = np.argsort(-np.array(row))[1:k_value + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)

    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    idx = 0
    results_dorsal = ppr(sim_graph, images_list, dorsal_list)
    results_palmar = ppr(sim_graph, images_list, palmar_list)
    final_results = {}

    for img in images_list_unlab:
        if results_dorsal[img] < results_palmar[img]:
            final_results[img] = "dorsal"
        else:
            final_results[img] = "palmar"

    actual_labels = fetch_actual_labels(images_list_unlab)
    print("Classification")
    no_correct = 0
    correctly_classified = []
    incorrectly_classified = []
    print("|   ImageId          | Prediction |  Actual |")
    for r in final_results:
        print("|   {} |   {}   |  {} |".format(r, final_results[r], actual_labels[r]))
        if final_results[r] == actual_labels[r]:
            correctly_classified.append(r)
            no_correct += 1
        else:
            incorrectly_classified.append(r)

    print("Correctly classified: {}\n".format(correctly_classified))
    print("InCorrectly classified: {}\n".format(incorrectly_classified))

    print("Classification Accuracy: {}".format(no_correct / len(images_list_unlab) * 100))
    print("Execution time: {} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()
