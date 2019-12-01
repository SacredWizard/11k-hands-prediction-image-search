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
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time
import numpy as np
from pandas import DataFrame
import pandas
# pandas.set_option('display.max_rows', 500)
# pandas.set_option('display.max_columns', 500)
# pandas.set_option('display.width', 1000)

def ppr(sim_graph, images_list, query_images, max_iter=500, alpha=0.85):
    sim_graph = sim_graph.T
    teleport_matrix = np.array([0 if img not in query_images else 1 for img in images_list]).reshape(len(images_list), 1)
    teleport_matrix = teleport_matrix / len(query_images)
    print(teleport_matrix)
    print(teleport_matrix.shape)
    uq_new = teleport_matrix
    uq_old = np.array((len(images_list), 1))
    iter = 0
    while iter < max_iter and not np.array_equal(uq_new, uq_old):
        uq_old = uq_new.copy()
        uq_new = alpha * np.matmul(sim_graph, uq_old) + (1 - alpha) * teleport_matrix
        iter += 1
    print("Iterations: {}".format(iter))
    print(uq_new)
    uq_new = uq_new.ravel()
    print(uq_new)
    # uq_new = uq_new[::-1].argsort(axis=0)
    a =(-uq_new).argsort()
    for i in a:
        print(images_list[i])

    # print(sort)
    # print(uq_new)








def main():
    """Main function for the script"""
    start = time.time()
    feature_extraction_model = "HOG"
    dimension_reduction_model = "PCA"
    k_value = 5
    dim_k_value = 40
    folder = "Dataset3/Labelled/Set2"
    dim_red = DimensionReduction(feature_extraction_model, dimension_reduction_model, dim_k_value, folder_metadata=folder,
                                 metadata_collection="labelled")
    features_list = np.array(dim_red.get_object_feature_matrix()['featureVector'].tolist())
    images_list = np.array(dim_red.get_object_feature_matrix()['imageId'])
    # eucl_dist = euclidean_distances(features_list)
    eucl_dist = cosine_similarity(features_list)
    # eucl_dist = map(lambda x: 1/ (1 + x), eucl_dist)
    pd = {}
    pd["imageId"] = images_list
    idx = 0
    for d in eucl_dist:
        pd[images_list[idx]] = d
        idx += 1

    df = DataFrame(pd)
    df = df.set_index("imageId")
    print(df)
    sim_graph = np.empty((0, len(eucl_dist)))
    # sim_matrix = np.empty((0, len(eucl_dist)))
    for row in eucl_dist:
        k_largest = np.argsort(-np.array(row))[1:k_value+1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
    #     sum_row = sum(sim_graph_row)
    #     # print(sim_row)
    #     sim_graph_row = list(map(lambda x: x / sum_row, sim_graph_row))
    #     # print(sim_graph_row)
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    #     sim_matrix = np.append(sim_matrix, np.array([sim_row]), axis=0)
    # print(sim_graph)
    print(sim_graph)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    print(sim_graph)
    idx = 0
    for img in images_list:
        df.loc[img] = sim_graph[idx]
        idx += 1
    # print(df)
    # print(sim_graph.tolist())
    print("\n\n")
    # print(sim_graph.T)
    # print(sim_graph[9][8])
    # ppr(sim_graph, images_list, ["Hand_0008333.jpg", "Hand_0006183.jpg", "Hand_0000074.jpg"])
    ppr(sim_graph, images_list, ["Hand_0003457.jpg", "Hand_0000074.jpg", "Hand_0005661.jpg"])
    print("Execution time: {} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()


 # print(sim_matrix)
    # for i in sim_graph:
    #     print(i)
    #     dist_max = (max(row))
    #
    #     def sim_score(x):
    #         # s = (1 - x / dist_max) * 100
    #         s = 1 / 1 + x
    #         return 0 if s == 1 else s
    #     sim_row = list(map(sim_score, row))