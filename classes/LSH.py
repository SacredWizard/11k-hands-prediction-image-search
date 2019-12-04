import math
import operator

import numpy as np

from classes.globalconstants import GlobalConstants
from utils.distancemeasure import euclidean
from utils.model import Model


class LSH:
    def __init__(self, layers=10, khash_count=10, w=1, image_ids=None, data=None):
        self.layers = layers
        self.w = w
        self.khash_count = khash_count
        self.data = data
        self.image_ids = image_ids

    def get_shape(self):
        return self.data.shape

    def get_l(self):
        return self.layers

    def get_k(self):
        return self.khash_count

    def create_index(self):
        l_hash, l_bucket, k_hash, k_bucket, image_hash, bucket = {}, {}, {}, {}, {}, {}
        images_count, dimensions = self.data.shape
        projections = np.zeros((self.khash_count, images_count))
        b = np.random.uniform(0, self.w, (self.khash_count, 1))
        self.w = (np.linalg.norm(b)/2)

        for layer in range(self.layers):
            k_hash, k_bucket = {}, {}
            b = np.random.uniform(0, self.w, (self.khash_count, 1))

            for kid in range(self.khash_count):
                vector = np.random.normal(0, 1, (1, dimensions))
                unit_vector = vector / np.linalg.norm(vector)
                image_hash, bucket = {}, {}

                for i in range(images_count):
                    projections[kid][i] = np.dot(self.data[i], unit_vector.T)
                    bucket_id = math.floor((projections[kid][i] + b[kid][0]) / self.w)
                    image_hash[self.image_ids[i]] = bucket_id
                    if bucket_id not in bucket:
                        bucket[bucket_id] = [self.image_ids[i]]
                    else:
                        bucket[bucket_id].append(self.image_ids[i])

                k_hash["K{}".format(kid)] = image_hash.copy()
                k_bucket["K{}".format(kid)] = bucket.copy()

            l_hash["L{}".format(layer)] = k_hash.copy()
            l_bucket["L{}".format(layer)] = k_bucket.copy()

        return l_hash, l_bucket

    def query(self, query_id, top):
        model = Model()
        constants = GlobalConstants()
        l_hash, l_bucket = model.load_model(constants.LSH_L_HASHES), model.load_model(constants.LSH_L_BUCKETS)
        choices_per_layer, choices_per_hash = [], []
        query_vector = self.data[self.image_ids.index(query_id)]
        imageid_distances = []

        for layer in range(self.layers):
            k_hash = l_hash["L{}".format(layer)]
            k_bucket = l_bucket["L{}".format(layer)]
            choices_per_hash = []

            for kid in range(self.khash_count):
                choices_per_hash.append(k_bucket["K{}".format(kid)].get(k_hash["K{}".format(kid)].get(query_id)))

            choices_per_layer.append(set.intersection(*map(set, choices_per_hash)))
        choices = set.union(*map(set, choices_per_layer))

        for image_id in choices:
            if image_id != query_id:
                vector = self.data[self.image_ids.index(image_id)]
                imageid_distances.append((image_id, euclidean(query_vector, vector)))

        imageid_distances = sorted(imageid_distances, key=operator.itemgetter(1))
        choices = [image for (image, distance) in imageid_distances]

        if len(choices) < top:
            # new_choices = search_neighbors()
            pass

        feat_vectors = {}
        print("Overall images: {}".format(len(choices)))
        for i in choices[:top]:
            feat_vectors[i] = self.data[self.image_ids.index(i)]
        return choices[:top], feat_vectors, query_vector
