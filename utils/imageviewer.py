"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np


def show_images(query_image, image_list):
    """Visualizer for the images"""
    f = plt.figure()

    no_images_per_row = 5
    no_of_lines = int(len(image_list) / no_images_per_row + 1)
    print(no_of_lines)

    f.add_subplot(no_of_lines, 5, 1)
    plt.imshow(io.imread(query_image))
    plt.axis('off')
    plt.title("Query Image")

    count = 2
    for r in image_list:
        f.add_subplot(no_of_lines, 5, count)
        plt.imshow(io.imread(os.path.join(r['path'])))
        plt.title(
            "{}\nScore: {}%".format(r['imageId'], round(r['score'], 3)))
        plt.axis('off')
        count = count + 1
    plt.show()

def show_feature_ls(data, feat_lat, title):
    plt.rcParams.update({'font.size': 10})
    """Visualizer for the images"""
    f = plt.figure(figsize=(20, 12))
    title_visualizer = ""
    for i in title:
        title_visualizer += i + ":" + str(title[i]) + "  "

    title_visualizer += "Feature Latent Semantics"
    f.suptitle(title_visualizer, fontsize=18)
    
    #extra credit for feature latent semantic
    image_list = []
    for k in feat_lat:
        k_list = [k.dot(data['featureVector'][i]) for i in range(len(data['featureVector']))]
        index = np.argsort(-np.array(k_list))[0]
        print(index)
        rec = dict()
        rec['imageId'] = data['imageId'][index]
        rec['path'] = data['path'][index]
        image_list.append(rec)

    no_images_per_row = 5
    no_of_lines = int(len(image_list) / no_images_per_row + 1)

    count = 1
    for r in image_list:
        f.add_subplot(no_of_lines, no_images_per_row, count)
        plt.imshow(io.imread(os.path.join(r['path'])))
        plt.title(
            "Latent semantic {} \n{}".format(count, r['imageId']))
        plt.axis('off')
        count = count + 1

    fig = plt.gcf()
    fig.set_size_inches((20, 12), forward=True)
    filename = "output/{}_feature_ls".format("_".join([str(i) for i in title.values()]))
    fig.savefig(filename, dpi=500)