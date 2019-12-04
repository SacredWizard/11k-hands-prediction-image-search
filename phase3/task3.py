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
from utils.inputhelper import get_input_k, get_input_folder, get_input_image_list
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time
import os
import numpy as np
from pandas import DataFrame
from utils.imageviewer import show_images_ppr


def ppr(sim_graph, images_list, query_images, max_iter=500, alpha=0.85):
    sim_graph = sim_graph.T
    teleport_matrix = np.array([0 if img not in query_images else 1 for img in images_list]).reshape(len(images_list), 1)
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
    a = (-uq_new).argsort()
    result = []
    rank = 1
    for i in a:
        res = {"imageId": images_list[i], "score": uq_new[i], "rank": rank}
        result.append(res)
        rank += 1
    return result


def sim_graph_from_sim_max(cos_sim, images_list, k_value):
    pd = {"imageId": images_list}
    idx = 0
    for d in cos_sim:
        pd[images_list[idx]] = d
        idx += 1

    df = DataFrame(pd)
    df = df.set_index("imageId")
    sim_graph = np.empty((0, len(cos_sim)))
    # sim_matrix = np.empty((0, len(eucl_dist)))
    for row in cos_sim:
        k_largest = np.argsort(-np.array(row))[1:k_value + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)

    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    return sim_graph


def main():
    """Main function for the script"""
    feature_extraction_model = "HOG"
    dimension_reduction_model = "PCA"
    k_value = get_input_k("k")
    K_value = get_input_k("K")
    folder = get_input_folder("Folder")
    dim_k_value = 40

    query_images = get_input_image_list(folder)
    start = time.time()
    dim_red = DimensionReduction(feature_extraction_model, dimension_reduction_model, dim_k_value, folder_metadata=folder,
                                 metadata_collection="labelled")
    obj_feat = dim_red.get_object_feature_matrix()
    features_list = np.array(obj_feat['featureVector'].tolist())
    images_list = np.array(obj_feat['imageId'])
    cos_sim = cosine_similarity(features_list)

    sim_graph = sim_graph_from_sim_max(cos_sim, images_list, k_value)
    results = ppr(sim_graph, images_list, query_images)
    results = results[:K_value]

    print("Top {} images from Personalized page Rank are:".format(K_value))
    for r in results:
        r["path"] = os.path.abspath(os.path.join(folder, r['imageId']))
        print(r)

    query_images_list = [os.path.abspath(os.path.join(folder, img)) for img in query_images]
    title = {"Model": "Personalized Page Rank", "k": k_value, "K": K_value}
    show_images_ppr(query_images_list, title, results)
    print("Execution time: {} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()
