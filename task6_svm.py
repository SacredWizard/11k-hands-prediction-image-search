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
import time
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, send_from_directory,url_for
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
    training_set = os.path.abspath('Dataset3\Labelled\Set2')
    test_set = os.path.abspath('Dataset3\\Unlabelled\Set 1')
    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)

    filename = "{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, str(k_value), os.path.basename(training_set))
    model = model_interact.load_model(filename=filename)
    if model is None:
        p2task1.save_model(dim_reduction, fea_ext_mod, dim_red_mod, str(k_value),folder=os.path.basename(training_set))
        model = model_interact.load_model(filename=filename)
    results = dim_reduction.find_m_similar_images(model, t_value, training_set, query_image, dist_func)
    return results

"""
Method to incorporate relevance feedback
"""
def rewrite_query(feedback):
    rel_similar_images = ["Hand_0000072.jpg","Hand_0000073.jpg","Hand_0000074.jpg","Hand_0000112.jpg"]
    random.shuffle(rel_similar_images)
    return rel_similar_images

def main():
    query_image = "Hand_0000072.jpg"
    # results = search_LSH(query_image)
    irrelevant=0
    relevant=0
    # similar_images = list(i['imageId'] for i in results)
    similar_images = ["Hand_0000072.jpg","Hand_0000073.jpg","Hand_0000074.jpg","Hand_0000112.jpg"]
    relevancefeedback.relevance_fdbk(os.path.abspath("Hands"),"SVM",query_image,similar_images)
    pass

if __name__ == "__main__":
    main()
    # app.run(port=5000, debug=True)
