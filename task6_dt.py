import pdb
import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
import utils.relevancefeedback as relevancefeedback
import numpy as np
import random as random
import operator
import utils.imageviewer as imgvwr
import phase2.task1 as p1task1
import phase2.task1 as p2task1
import phase3.task1 as p3task1
import time
import warnings
#from task4_svm import SupportVectorMachine, gaussian_kernel

warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, send_from_directory, url_for

app = Flask(__name__)

model_interact = Model()

"""
Dummy. TO be replaced by actual LSH method
"""


def search_LSH(query_image):
    fea_ext_mod = "HOG"
    dim_red_mod = "SVD"
    dist_func = "euclidean"
    k_value = 30
    t_value = 20
    training_set = os.path.abspath('Dataset3/Labelled/Set2')
    test_set = os.path.abspath('Dataset3/Unlabelled/Set 1')
    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)

    filename = "{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, str(k_value), os.path.basename(training_set))
    model = model_interact.load_model(filename=filename)
    if model is None:
        p2task1.save_model(dim_reduction, fea_ext_mod, dim_red_mod, str(k_value), folder=os.path.basename(training_set))
        model = model_interact.load_model(filename=filename)
    results = dim_reduction.find_m_similar_images(model, t_value, training_set, query_image, dist_func)
    return results


"""
Method to incorporate relevance feedback
"""


def rewrite_query(feedback):
    # pdb.set_trace()
    global feedback_imgs_g, feedback_vals_g, similar_images_g, similar_image_vectors_g

    feedback = {'Hand_0000072.jpg': '1', 'Hand_0000073.jpg': '-1', 'Hand_0000074.jpg': '1', 'Hand_0000112.jpg': '-1'}
    # Add SVM based relevance feedback function
    # """    clf = SupportVectorMachine(gaussian_kernel, C=500)
    #     feedback_imgs = list(feedback.keys())
    #     feedback_vals = list(feedback.values())
    #     x_train=[]
    #     y_train=[]
    #     for i in range(similar_images_g):
    #         for j in range(feedback_imgs_g):
    #             if similar_images_g[i] == feedback_imgs_g[j]:
    #                 x_train.append(similar_image_vectors_g[i])
    #                 y_train.append(feedback_vals_g[j])
    #     clf.fit(x_train, y_train)
    #     x_test = similar_images_g
    #     image_dist_SV = clf.project(x_test)
    #     rel_similar_images = sorted(image_dist_SV,reversed=True)
    #     """
    rel_similar_images = feedback
    return rel_similar_images


similar_images_g = []
similar_image_vectors_g = []
feedback_imgs_g = []
feedback_vals_g = []


def main():
    query_image = "Hand_0000072.jpg"
    # results = search_LSH(query_image)
    results = ['Hand_0000072.jpg', 'Hand_0000073.jpg', 'Hand_0000074.jpg', 'Hand_0000112.jpg']
    global similar_images_g, similar_image_vectors_g
    # similar_images_g = list(i['imageId'] for i in results)
    # similar_image_vectors_g = list(i['reducedDimensions'] for i in results)

    similar_images_g = results

    # rewrite_query(None)
    relevancefeedback.relevance_fdbk(os.path.abspath("Hands"), "SVM", query_image, similar_images_g)
    pass


if __name__ == "__main__":
    main()