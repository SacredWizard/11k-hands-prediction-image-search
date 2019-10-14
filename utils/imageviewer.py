"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import os
import matplotlib.pyplot as plt
from skimage import io


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
            "{}\nDist: {}".format(r['imageId'], round(r['dist'], 5)))
        plt.axis('off')
        count = count + 1
    plt.show()
