import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from classes.dimensionreduction import DimensionReduction
from classes.LSH import LSH
from utils.model import Model
import utils.relevancefeedback as relevancefeedback
from utils.inputhelper import get_input_image
import numpy as np
import random as random
import operator
import utils.imageviewer as imgvwr
import phase2.task1 as p1task1
import phase2.task1 as p2task1
import phase3.task1 as p3task1
import task5 as p3task5
import time
import warnings
from task4_svm import SupportVectorMachine,gaussian_kernel
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, send_from_directory,url_for
app = Flask(__name__)
model_interact = Model()
#global vars to retain feedback and initial set of images returned from LSH
similar_images_g = []
similar_image_vectors_g = []
feedback_imgs_g = []
feedback_vals_g = []

def get_LSH_results(query_image):
    lsh = LSH()
    global similar_images_g,similar_image_vectors_g
    similar_images_g,similar_image_vectors_g, query_image_vector = p3task5.task5b(query_image, 20)
    # print(len(similar_images_g),len(similar_image_vectors_g))
    return similar_images_g,similar_image_vectors_g, query_image_vector

def get_training_set(feedback_imgs,feedback_vals):
    x_train=[]
    y_train=[]
    global similar_images_g,similar_image_vectors_g
    global feedback_imgs_g, feedback_vals_g
    # update overall feedback
    for img,val in zip(feedback_imgs,feedback_vals):
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
    return x_train,y_train

"""
Method to incorporate relevance feedback
"""
def rerank_results(feedback,similar_images,similar_image_vectors):
    global feedback_imgs_g,feedback_vals_g,similar_images_g,similar_image_vectors_g
    similar_images_g = similar_images
    similar_image_vectors_g = similar_image_vectors
    # feedback = {'Hand_0000071.jpg': '1', 'Hand_0000073.jpg': '1'}
    # feedback = {'Hand_0000071.jpg': '-1', 'Hand_0000073.jpg': '-1','Hand_0000074.jpg': '1', 'Hand_0000070.jpg': '-1', 'Hand_0000075.jpg': '-1', 'Hand_0000078.jpg': '1', 'Hand_0009068.jpg': '1', 'Hand_0007705.jpg': '1', 'Hand_0000076.jpg': '-1', 'Hand_0007228.jpg': '-1', 'Hand_0009162.jpg': '-1', 'Hand_0009163.jpg': '-1'}
    # Add SVM based relevance feedback function
    clf = SupportVectorMachine(gaussian_kernel, C=500)
    feedback_imgs = list(feedback.keys())
    feedback_vals = list(feedback.values())
    x_train,y_train = get_training_set(feedback_imgs,feedback_vals)
    # print(len(similar_images_g),len(similar_image_vectors_g),len(feedback_imgs_g),len(feedback_vals_g))
    clf.fit(np.array(x_train), np.array(y_train))
    x_test = similar_image_vectors_g
    image_dist_SVM = clf.project(list(x_test.values()))
    image_dist_SVM_index = [i[0] for i in sorted(enumerate(image_dist_SVM), key=lambda x:x[1], reverse=True)]
    rel_similar_images = [list(similar_image_vectors_g.keys())[index] for index in image_dist_SVM_index] 
    # list(similar_image_vectors_g.keys())[image_dist_SVM_index]
    return rel_similar_images

def main():
    query_image = get_input_image("Hands")
    similar_images,img_vectors, query_image_vector = get_LSH_results(query_image)
    # while True:
    #     rerank_results(None)
    relevancefeedback.relevance_fdbk("SVM",query_image,similar_images,img_vectors, query_image_vector)
    pass

if __name__ == "__main__":
    main()
