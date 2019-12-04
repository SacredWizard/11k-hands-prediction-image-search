"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pandas as pd


def show_images_ppr(query_images, title, image_list):
    """Visualizer for the images"""
    f = plt.figure(figsize=(20, 12))
    title_visualizer = ""
    for i in title:
        title_visualizer += i + ":" + str(title[i]) + "  "

    f.suptitle(title_visualizer, fontsize=18)

    no_images_per_row = 5
    no_of_lines = int(len(image_list) / no_images_per_row + 1)
    idx = 1
    # Query Images
    for img in query_images:
        f.add_subplot(no_of_lines, 5, idx)
        plt.imshow(io.imread(img))
        plt.axis('off')
        plt.title("Q{} : {}".format(idx, os.path.basename(img)))
        idx = idx + 1

    # Results
    for r in image_list:
        f.add_subplot(no_of_lines, 5, idx)
        plt.imshow(io.imread(r['path']))
        plt.title(
            "{}\nRank: {}".format(r['imageId'], round(r['score'], 3)))
        plt.axis('off')
        idx = idx + 1

    fig = plt.gcf()
    fig.set_size_inches((20, 12), forward=True)
    filename = "output/{}".format("_".join([str(i) for i in title.values()]) + "_")
    print("Visualizer Image saved to: {}.png".format(filename))
    fig.savefig(filename, dpi=500)
    plt.show()


def show_images(query_image, image_list, title):
    """Visualizer for the images"""
    f = plt.figure(figsize=(20, 12))
    title_visualizer = ""
    for i in title:
        title_visualizer += i + ":" + str(title[i]) + "  "

    f.suptitle(title_visualizer, fontsize=18)

    no_images_per_row = 5
    no_of_lines = int(len(image_list) / no_images_per_row + 1)

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

    fig = plt.gcf()
    fig.set_size_inches((20, 12), forward=True)
    filename = "output/{}".format("_".join([str(i) for i in title.values()]) + "_" + query_image.split(os.path.sep)[-1].
                                  strip(".jpg"))
    print("Visualizer Image saved to: {}.png".format(filename))
    fig.savefig(filename, dpi=500)
    plt.show()

def show_data_ls(data, data_tw, title):
    """
    :param data_tw: term weight pairs for data latent semantics
    :param title: Image Title
    :return: Nothing!
    """
    plt.rcParams.update({'font.size': 6})

    f = plt.figure(figsize=(20, 12))
    title_visualizer = ""
    for i in title:
        title_visualizer += i + ":" + str(title[i]) + "  "

    title_visualizer += "Data Latent Semantics"
    f.suptitle(title_visualizer, fontsize=18)
    
    #extra credit for data latent semantic
    k = len(data_tw)
    no_images_per_row = 10
    no_of_lines = k

    count = 1
    for i in range(k):
        for j in range(no_images_per_row + 1):
            if j == 0:
                f.add_subplot(no_of_lines, no_images_per_row + 1, count)
                plt.title("LS {}".format(i+1), fontsize=9)
                plt.axis('off')
                count = count + 1
            else:
                f.add_subplot(no_of_lines, no_images_per_row + 1, count)
                imageId = data_tw[i][j-1][0]
                score = data_tw[i][j-1][1]
                unique_index = pd.Index(data['imageId'])
                index = unique_index.get_loc(imageId)
                path = data['path'][index]
                plt.imshow(io.imread(os.path.join(path)))
                plt.title("{}\nScore:{}".format(imageId, round(score, 3)))
                plt.axis('off')
                count = count + 1

    fig = plt.gcf()
    fig.set_size_inches((20, 12), forward=True)
    filename = os.path.join(os.path.abspath("output"),"{}_data_ls".format("_".join([str(i) for i in title.values()])))
    print("Visualizer Image for data latent semantics saved to: {}.png".format(filename))
    fig.savefig(filename, dpi=500)

def show_feature_ls(data, feat_lat, title):
    """
    :param data: data containg the latent semantics
    :param feat_lat: Feature Latent semantics
    :param title: Image Title
    :return: Nothing!
    """
    plt.rcParams.update({'font.size': 7})

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
    filename = os.path.join(os.path.abspath("output"),"{}_feature_ls".format("_".join([str(i) for i in title.values()])))
    print("Visualizer Image for feature latent semantics saved to: {}.png".format(filename))
    fig.savefig(filename, dpi=500)


def show_subjectwise_images(given_subject_id, given_subject_images, subjects_with_scores, similar_subjects_images, fig_filename):
    # array of sub-plotss
    maximages = 5
    nrows, ncols = len(similar_subjects_images)+1, maximages
    figsize = [10,10]     # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    # max images to display per subject
    # plot images on each sub-plot
    
    ax[0][0].text(x = -1.0, y = 0.5, s = "Query ID : {0}".format(given_subject_id), rotation = 0,
                    horizontalalignment='center', verticalalignment='center', transform=ax[0][0].transAxes)
    for i,image in enumerate(given_subject_images):
        ax_subplot = (ax[0][i])
        ax_subplot.set_xlabel("xlabel")
        ax_subplot.imshow(io.imread(image))
        ax_subplot.set_title("{0}".format(str(os.path.basename(image))))
        ax_subplot.axis('off')

    for i,images_for_subject in enumerate(similar_subjects_images):
        # add subject ID and similarity score
        ax[i+1][0].text(x = -1.0, y = 0.5, s = "ID : {0}\n Score : {1}%".format(subjects_with_scores[i][0],
                        round(subjects_with_scores[i][1],2)), rotation = 0, horizontalalignment='center',
                        verticalalignment='center', transform=ax[i+1][0].transAxes)
        # populate images for each subject
        for j,image in enumerate(images_for_subject):
            ax_subplot = (ax[i+1][j])
            ax_subplot.set_xlabel("xlabel")
            ax_subplot.imshow(io.imread(image))
            ax_subplot.set_title("{0}".format(str(os.path.basename(image))))
            ax_subplot.axis('off')
            if j >= maximages-1:
                break
        # turn off axis and markings
        while j < maximages:
            ax_subplot = (ax[i+1][j])
            ax_subplot.axis('off')
            j+=1

    plt.tight_layout(True)
    save_fig = plt.gcf()
    save_fig.savefig(fig_filename)
    plt.show()
    
