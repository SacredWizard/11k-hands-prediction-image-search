import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
from classes.LSH import LSH
from utils.model import Model
import utils.relevancefeedback as relevancefeedback
from utils.inputhelper import get_input_image, get_input_k
import task5 as p3task5
import warnings
from task4_dt import DecisionTree
from knn import KNN

warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, send_from_directory, url_for

app = Flask(__name__)
model_interact = Model()
# global vars to retain feedback and initial set of images returned from LSH
similar_images_g = []
similar_image_vectors_g = []
feedback_imgs_g = []
feedback_vals_g = []


def get_LSH_results(query_image, no_images):
    lsh = LSH()
    global similar_images_g, similar_image_vectors_g
    similar_images_g, similar_image_vectors_g, query_image_vector = p3task5.task5b(query_image, no_images)
    # print(len(similar_images_g),len(similar_image_vectors_g))
    return similar_images_g, similar_image_vectors_g, query_image_vector


def get_training_set(feedback_imgs, feedback_vals):
    x_train = []
    y_train = []
    global similar_images_g, similar_image_vectors_g
    global feedback_imgs_g, feedback_vals_g
    # update overall feedback
    for img, val in zip(feedback_imgs, feedback_vals):
        # if user changes their feedback later
        if img in feedback_imgs_g:
            index = feedback_imgs_g.index(img)
            feedback_vals_g[index] = int(val)
        # new feedback
        else:
            feedback_imgs_g.append(img)
            feedback_vals_g.append(int(val))

    for i in range(len(similar_images_g)):
        for j in range(len(feedback_imgs)):
            if similar_images_g[i] == feedback_imgs[j]:
                x_train.append(similar_image_vectors_g[similar_images_g[i]])
                y_train.append(feedback_vals_g[j])
    return x_train, y_train


"""
Method to incorporate relevance feedback
"""


def rerank_results(feedback, similar_images, similar_image_vectors, query_image_vector):
    global feedback_imgs_g, feedback_vals_g, similar_images_g, similar_image_vectors_g
    similar_images_g = similar_images
    similar_image_vectors_g = similar_image_vectors

    # Add DT based relevance feedback function
    clf = DecisionTree()
    feedback_imgs = list(feedback.keys())

    feedback_vals = list(feedback.values())
    x_train_old, y_train = get_training_set(feedback_imgs, feedback_vals)
    x_train = []
    for i in x_train_old:
        j = i.tolist()
        x_train.append(j)

    clf.fit(x_train, y_train)
    # x_test = similar_image_vectors_g.values()
    x_test = []
    for i in similar_image_vectors_g.values():
        j = i.tolist()
        x_test.append(j)
    print(x_test)

    predictions = clf.predict(x_test)
    #relevant images
    indices_rel = [i for i, x in enumerate(predictions) if x == 1]
    x_train_knn_rel = []
    rel_len = len(indices_rel)
    for i in indices_rel:
        x_train_knn_rel.append(x_test[i])
    knn = KNN(rel_len)
    knn.fit(x_train_knn_rel)
    neighbours_rel = knn.get_neighbours([query_image_vector])
    #irrelevant images
    indices_ir = [i for i, x in enumerate(predictions) if x == -1]
    x_train_knn_ir = []
    ir_len = len(indices_ir)
    for i in indices_ir:
        x_train_knn_ir.append(x_test[i])
    knn = KNN(ir_len)
    knn.fit(x_train_knn_ir)
    neighbours_ir = knn.get_neighbours([query_image_vector])
    ranked_indices = []
    ranked_indices.extend(neighbours_rel)
    ranked_indices.extend(neighbours_ir)
    rel_similar_images = [list(similar_image_vectors_g.keys())[index] for index in ranked_indices]
    return rel_similar_images


def main():
    query_image = get_input_image("Hands")
    no_images = get_input_k("t")
    similar_images, img_vectors, query_image_vector = get_LSH_results(query_image, no_images)
    # while True:
    #     rerank_results(None)
    relevancefeedback.relevance_fdbk("DT", query_image, similar_images, img_vectors, query_image_vector)
    pass


if __name__ == "__main__":
    main()
