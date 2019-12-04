import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])
from classes.dimensionreduction import DimensionReduction
from classes.LSH import LSH
from utils.model import Model
from utils.inputhelper import get_input_image
import utils.relevancefeedback as relevancefeedback
from sklearn.metrics.pairwise import cosine_similarity
from phase3.task3 import ppr, sim_graph_from_sim_max
import numpy as np
import random as random
import operator
import utils.imageviewer as imgvwr
import phase2.task1 as p1task1
import phase2.task1 as p2task1
import phase3.task1 as p3task1
import task5 as p3task5
import sys
import time
import warnings
from task4_svm import SupportVectorMachine, gaussian_kernel

warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, send_from_directory, url_for

app = Flask(__name__)
model_interact = Model()
# global vars to retain feedback and initial set of images returned from LSH
similar_images_g = []
similar_image_vectors_g = []
feedback_imgs_g = []
feedback_vals_g = []


def get_LSH_results(query_image):
    lsh = LSH()
    print(query_image)
    global similar_images_g, similar_image_vectors_g
    similar_images_g, similar_image_vectors_g, query_image_vector = p3task5.task5b(query_image, 20)
    # print(len(similar_images_g),len(similar_image_vectors_g))
    return similar_images_g, similar_image_vectors_g, query_image_vector

"""
Method to incorporate relevance feedback
"""


def rerank_results(feedback, similar_images, similar_image_vectors):
    global feedback_imgs_g, feedback_vals_g, similar_images_g, similar_image_vectors_g
    similar_images_g = similar_images
    similar_image_vectors_g = similar_image_vectors
    # Add SVM based relevance feedback function
    print(feedback)
    images_list = list(feedback.keys())
    relevant_images = []
    images_list = []
    feature_vectors = []
    for image in similar_image_vectors:
        images_list.append(image)
        feature_vectors.append(similar_image_vectors[image])
        if feedback.get(image, 0) == "1":
            relevant_images.append(image)
    feature_vectors = np.array(feature_vectors)
    cos_sim = cosine_similarity(feature_vectors)
    sim_graph = sim_graph_from_sim_max(cos_sim, images_list, 5)
    results = ppr(sim_graph, images_list, relevant_images)
    rel_similar_images = []
    for img in results:
        rel_similar_images.append(img)
    return rel_similar_images


def main():
    query_image = get_input_image("Hands")
    # query_image = get_input_image()
    similar_images, img_vectors, query_image_vector = get_LSH_results(query_image)
    relevancefeedback.relevance_fdbk("PPR", query_image,similar_images,img_vectors, query_image_vector)
    # pass


if __name__ == "__main__":
    main()
