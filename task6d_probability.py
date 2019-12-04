import math
import operator
import os

import numpy as np

import task5
import utils.relevancefeedback as relevancefeedback
from utils.model import Model
from utils.inputhelper import get_input_image, get_input_k

feedback_metadata_obj = {}
user_relevant_images = []
user_irrelevant_images = []


def rerank_results(data):
    model = Model()
    for k in data.keys():
        if data[k] == 1:
            user_relevant_images.append(k)
            if k in user_irrelevant_images:
                user_irrelevant_images.remove(k)
        else:
            user_irrelevant_images.append(k)
            if k in user_relevant_images:
                user_relevant_images.remove(k)
    imageids = model.load_model('imageids')
    feat_vectors = model.load_model('feat_vectors')
    bin_matrix = model.load_model('bin_matrix')
    imageid_index = model.load_model('imageid_index')
    feature_index = model.load_model('feature_index')
    get_relevance(feat_vectors.keys(), bin_matrix, imageid_index, feature_index, feat_vectors)


def get_relevance(imageids, bin_matrix, imageid_index, feature_index, feat_vectors):
    similar_image = {}
    for i in imageids:
        similar_image[i] = get_image_similarity(i, imageid_index, feature_index, feat_vectors, bin_matrix)

    image_relevance = []
    for image, feature in sorted(similar_image.items(), key=operator.itemgetter(1), reverse=True):
        image_relevance.append(image)
    return image_relevance


def get_feedback_metadata(feature_index, imageid_index, bin_matrix):
    r = len(user_relevant_images)
    n = r + len(user_irrelevant_images)

    count = 0
    for image in user_relevant_images:
        img_index = imageid_index[image]
        if bin_matrix[img_index][feature_index] == 1:
            count += 1

    ri = count

    for image in user_irrelevant_images:
        img_index = imageid_index[image]
        if bin_matrix[img_index][feature_index] == 1:
            count += 1

    ni = count
    pi = (ri + 0.5) / float(r + 1)
    ui = (ni - ri + 0.5) / float(n - r + 1)
    return pi, ui


def get_image_similarity(imageid, imageid_index, feature_index, feat_vectors, bin_matrix):
    score = 0

    feature_vec = bin_matrix[imageid_index[imageid]]

    for feat in feature_index:
        if feat in feedback_metadata_obj.keys():
            (pi, ui) = feedback_metadata_obj[feat]
        else:
            (pi, ui) = get_feedback_metadata(feature_index[feat], imageid_index, bin_matrix)
            feedback_metadata_obj[feat] = (pi, ui)
        t = feature_vec[feature_index[feat]] * (math.log((pi * (1 - ui)) / (ui * (1 - pi))))
        score += t

    return score


def get_binary_matrix(feat_vectors):
    matrix = np.asarray([list(feat_vectors[k]) for k in feat_vectors])
    median = np.median(matrix)
    matrix[matrix > median] = 1
    matrix[matrix < median] = 0
    return matrix


def preprocess(feat_vectors):
    count = 0
    imageid_index, feature_index = {}, {}
    for k in feat_vectors.keys():
        imageid_index[k] = count
        feature_index[str(feat_vectors[k])] = count
        count += 1
    return imageid_index, feature_index


def get_probability_revelance_feedback(query_id, no_images):
    imageids, feat_vectors = task5.task5b(query_id, no_images)
    model = Model()
    bin_matrix = get_binary_matrix(feat_vectors)
    imageid_index, feature_index = preprocess(feat_vectors)
    model.save_model(imageids, 'imageids')
    model.save_model(feat_vectors, 'feat_vectors')
    model.save_model(bin_matrix, 'bin_matrix')
    model.save_model(imageid_index, 'imageid_index')
    model.save_model(feature_index, 'feature_index')
    relevancefeedback.relevance_fdbk(os.path.abspath("Hands"), "PROBABILITY", query_id,
                                     get_relevance(feat_vectors.keys(), bin_matrix, imageid_index, feature_index,
                                                   feat_vectors))


def main():
    query = get_input_image("Hands")
    no_images = get_input_k("t")
    get_probability_revelance_feedback(query, no_images)


if __name__ == '__main__':
    main()

