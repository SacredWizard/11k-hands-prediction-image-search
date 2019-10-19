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

def show_images(query_image, image_list, title):
    """Visualizer for the images"""
    f = plt.figure()
    f.suptitle(title, fontsize=20)

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

def show_subjectwise_images(subjects_with_scores, similar_subjects_images):
    # array of sub-plotss
    nrows, ncols = len(similar_subjects_images), (max(len(l) for l in similar_subjects_images))
    figsize = [10,10]     # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    # plot images on each sub-plot
    for i,images_for_subject in enumerate(similar_subjects_images):
        # add subject ID and similarity score
        ax[i][0].text(x = -1.0, y = 0.5, s = "ID : {0}\n Score : {1}%".format(subjects_with_scores[i][0],
                        round(subjects_with_scores[i][1],2)), rotation = 0, horizontalalignment='center',
                        verticalalignment='center', transform=ax[i][0].transAxes)
        # populate images for each subject
        for j,image in enumerate(images_for_subject):
            ax_subplot = (ax[i][j])
            ax_subplot.set_xlabel("xlabel")
            ax_subplot.imshow(io.imread(image))
            ax_subplot.set_title("{0}".format(str(os.path.basename(image))))
            ax_subplot.axis('off')
        # turn off axis and markings  
        while j < ncols:
            ax_subplot = (ax[i][j])
            ax_subplot.axis('off')
            j+=1

    plt.tight_layout(True)
    plt.show()
