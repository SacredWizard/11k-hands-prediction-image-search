"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import os
import matplotlib.pyplot as plt
from skimage import io


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
    filename = "output/{}".format("_".join([str(i) for i in title.values()]) + "_" + query_image.split("/")[-1].
                                  strip(".jpg"))
    fig.savefig(filename, dpi=500)
    plt.show()

