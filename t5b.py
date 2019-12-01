import collections
import operator
import pickle

import numpy as np

from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper
from utils import model

images = []


##############################################
# Load LSH index structure from pickle file
##############################################
def load_LSH_pickle():
    filename = "models/LSH_hashes"
    infile = open(filename, 'rb')
    LSH_hash_table = pickle.load(infile)
    infile.close()

    filename = "models/LSH_buckets"
    infile = open(filename, 'rb')
    LSH_bucket = pickle.load(infile)
    infile.close()

    return LSH_hash_table, LSH_bucket


##################################################
# Find intersection of candidates within a Layer
##################################################
def find_intersection(candidates_per_hash, K):
    set_list = []

    for i in range(K):
        set_list.append(set(candidates_per_hash[i]))

    candidates_per_layer = list(set.intersection(*set_list))
    # print("Candidates per layer:{}".format(candidates_per_layer))
    return candidates_per_layer


######################################################
# Return Final candidates by applying union operation
# candidates found in each layer
######################################################
def final_candidates(candidates_per_layer, L):
    set_list = []
    for i in range(L):
        set_list.append(set(candidates_per_layer[i]))

    final_candidates = list(set.union(*set_list))

    return final_candidates


#######################################################
# Return euclidean distance between two vectors a and b
#######################################################
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


#######################################################
# Find distances between images in candidate list and
# query image id
######################################################
def find_distances(candidates, queryID, global_image_ids, obj_feature_mat):
    query_ind = global_image_ids.index(query_ID)
    query_vec = obj_feature_mat[query_ind]
    imageid_distance_tuples = []

    for img_id in candidates:
        if img_id != queryID:
            ind = global_image_ids.index(img_id)
            vec = obj_feature_mat[ind]
            d = euclidean_distance(vec, query_vec)
            imageid_distance_tuples.append((img_id, d))

    imageid_distance_tuples = sorted(imageid_distance_tuples, key=operator.itemgetter(1))

    return imageid_distance_tuples


###############################################################
# Search candidates in adjoining buckets of the query image id
# Only when required number of candidates is not met
###############################################################
def search_neighbor_buckets(LSH_hash_table, LSH_bucket, queryID):
    candidates_per_layer_nn = []
    candidates_per_hash_nn = []
    nn = 2
    for i in range(L):
        K_hashes = LSH_hash_table["L" + str(i)]
        K_buckets = LSH_bucket["L" + str(i)]

        candidates_per_hash_nn.clear()
        new_ind = []
        for j in range(K):
            new_ind.clear()
            bucketID = K_hashes["K" + str(j)].get(query_ID)
            # bucket_list = K_buckets["K" + str(j)].keys()
            ordered_dict = collections.OrderedDict(sorted(K_buckets["K" + str(j)].items()))
            bucket_list = list(ordered_dict.keys())

            ind = bucket_list.index(bucketID)

            if len(bucket_list) > nn:
                if (ind < nn):
                    count = nn - ind
                    new_ind = list(range(ind))
                    new_ind.extend(list(range(ind + 1, ind + count + 1)))

                elif ((len(bucket_list) - ind) <= nn):
                    count = nn - len(bucket_list) - ind - 1
                    new_ind = list(range(ind + 1, len(bucket_list)))
                    new_ind.extend(list(range(ind - count, ind)))
                else:
                    new_ind = list(range(ind - nn, ind))
                    new_ind.extend(list(range(ind + 1, ind + nn + 1)))
            if len(bucket_list) <= nn:
                new_ind = list(range(len(bucket_list)))
                new_ind.remove(ind)

            imglist = K_buckets["K" + str(j)].get(bucketID)
            for index in new_ind:
                imglist.extend(K_buckets["K" + str(j)].get(bucket_list[index]))

            candidates_per_hash_nn.append(imglist)

        candidates_per_layer_nn.append(find_intersection(candidates_per_hash, K))

    candidates_list = final_candidates(candidates_per_layer_nn, L)
    return candidates_list


def img_ids():
    constants = GlobalConstants()
    image_names = list(MongoWrapper(constants.Mongo().DB_NAME).find(constants.CM.lower(), {}, {"_id": 0, "imageId": 1}))
    # print(image_names)
    return list(map(lambda x: x['imageId'], image_names))

##############################################
# Execution Starts Here
##############################################

# LSH_hash_table, LSH_bucket = load_LSH_pickle()
LSH_hash_table = model.Model().load_model("L_hashes")
LSH_bucket = model.Model().load_model("L_buckets")

# query_ID = sys.argv[1]
# t = int(sys.argv[2])
t = 5000
query_ID = "Hand_0000003.jpg"

K_hash = LSH_hash_table["L0"]
L = len(LSH_hash_table.keys())
K = len(K_hash.keys())

candidates_per_layer = []
candidates_per_hash = []
filename = "img_dict"
# infile = open(filename, 'rb')
# img_dict = pickle.load(infile)
# infile.close()

# global_image_ids = img_dict["imglist"]
global_image_ids = img_ids()
# obj_feature_mat = img_dict["obj_feature_mat"]
obj_feature_mat = model.Model().load_model("lsh_nmf_w")

for i in range(L):
    K_hashes = LSH_hash_table["L" + str(i)]
    K_buckets = LSH_bucket["L" + str(i)]

    # for k in K_buckets:
    #     for nm in K_buckets[k]:
    #         print(len(K_buckets[k][nm]))
    #     # print(K_buckets[k])
    # exit(0)

    candidates_per_hash.clear()
    for j in range(K):
        bucketID = K_hashes["K" + str(j)].get(query_ID)
        # print(bucketID)
        imglist = K_buckets["K" + str(j)].get(bucketID)
        # print(K_buckets["K" + str(j)].keys())
        candidates_per_hash.append(imglist)
        # print("L:{},K:{}, can_per_hash:{}".format(i, j, imglist))

    candidates_per_layer.append(find_intersection(candidates_per_hash, K))

candidates = final_candidates(candidates_per_layer, L)

imageid_distance_tuples = find_distances(candidates, query_ID, global_image_ids, obj_feature_mat)

candidate_list = [img for (img, d) in imageid_distance_tuples]
if len(candidate_list) < t:
    candidates_new = search_neighbor_buckets(LSH_hash_table, LSH_bucket, query_ID)
    imageid_distance_tuples = find_distances(candidates_new, query_ID, global_image_ids, obj_feature_mat)
    candidate_list_new = [img for (img, d) in imageid_distance_tuples]
    candidate_list = candidate_list_new

images.append(str(query_ID) + '.jpg')
print(candidate_list[:t])
for each in candidate_list[:t]:
    images.append(str(each) + '.jpg')

print("Total number of unique images considered: {}".format(len(candidate_list)))



