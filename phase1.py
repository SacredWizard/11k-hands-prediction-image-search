from skimage.feature import local_binary_pattern, hog
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
from PIL import Image
import sys
import warnings
import cv2
import os
import pymongo
import ast
from math import*
import operator
import __future__

# author: Baani Khurana

# create connection to mongodb client
client = pymongo.MongoClient("mongodb://localhost:27017")

# connect to phase 1 database
db = client["phase1"]
    
# takes image file path as input and returns lbp vector for image
def implement_lbp(image_id):

    # parameter settings for local binary pattern calculation
    numPoints = 8
    radius = 1

    # name of the file is the image_id pathway
    filename = image_id

    # reads in the image
    image = imread(filename)

    # converts image to grayscale
    image = rgb2gray(image)

    # dimensions for 100x100 windows for the image
    window_rows = 100 
    window_columns = 100

    lbp_list = []

    # splits the 1600x1200 image into 192 blocks of 100x100 pixels and calculates lbp vector for each block
    for row in range(0, image.shape[0], window_rows):
        for column in range(0, image.shape[1], window_columns):
            window = image[row:row+window_rows,column:column+window_columns]
            lbp = local_binary_pattern(window, numPoints, radius, 'uniform')

            # appends the lbp vector for each window into a concatenated list
            lbp_list.append(lbp.tolist())
    
    lbp_list = np.asarray(lbp_list).ravel()
    lbp_list = lbp_list.tolist()

    return lbp_list

# takes image file path as input and returns hog vector for image
def implement_hog(image_id):
    
    # name of the file is the image_id pathway
    fileName = image_id

    # reads in the image
    img = cv2.imread(fileName)

    # downscales images 1 per 10 rows so 10% which means 1600x1200 to 160x120
    downscaled_image = rescale(img, 0.10, anti_aliasing=False)
   
    # calculates the hog vector for the image with 9 orientation bins, 8x8 cell size, block size of 2, and L2 norm.
    hog_image = hog(downscaled_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2', multichannel=True, feature_vector=True)
   
    # converts hog vector from a numpy array to a list
    hog_image = hog_image.tolist()

    return hog_image

# task1 method takes an image ID and feature model, extracts, and prints the feature vector in an output file.
def task1(image_id, feature_descriptor):
    if (feature_descriptor == "lbp"):
        output = implement_lbp(image_id)
        f = open("lbp_output.txt", "w+")
    elif(feature_descriptor == "hog"):
        output = implement_hog(image_id)
        f = open("hog_output.txt", "w+")
    
    # writes output to a file
    f.write(str(output))

# task2 method takes in a folder with images, extracts, and stores the feature models for all the images in mongoDB.
def task2(directory, feature_descriptor):        

    #iterates through each image in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_id = os.path.join(directory, filename)
            if (feature_descriptor == "lbp"):
                output = implement_lbp(image_id)
            elif (feature_descriptor == "hog"):
                output = implement_hog(image_id)

            # inserts the feature vector into the database with corresponding image_id
            collection = db[feature_descriptor]
            record = {"image_id": image_id, feature_descriptor: str(output)}
            collection.insert_one(record)

# method takes two vectors and calculates euclidean distance
def euclidean_distance(x, y):
    diff = x - y
    dist = np.sqrt(np.sum(np.square(diff)))
    return dist

# task3 method takes an image_id, a feature model, and value k and finds top k most similar images based on euclidean distance
def task3(image_id, feature_descriptor, k):

    # converts k to an integer from a string input
    k = int(k)

    # finds the query image in the database and retrieves its feature vector
    collection = db[feature_descriptor]
    image = collection.find_one({"image_id": image_id})
    img_string = image[feature_descriptor]
    img_concatenated = []
    img_value = [float(x) for x in img_string.strip("[]").split(",")]
    img_concatenated.append(img_value)
    img_nparray = np.asarray(img_concatenated)

    # initializing a dictionary for images and the distance
    my_images={}

    # iterates through the entire database and calculates the euclidean distance for each image
    for obj in collection.find():
        obj_id = obj['image_id']
        obj_string = obj[feature_descriptor]
        obj_concatenated = []
        obj_value = [float(x) for x in obj_string.strip("[]").split(",")]
        obj_concatenated.append(obj_value)
        obj_nparray = np.asarray(obj_concatenated)

        # calculates the euclidean distance
        distance = euclidean_distance(img_nparray, obj_nparray)
        
        # stores the distances in a dictionary
        my_images[obj_id] = distance
    
    # sorts the distances to detect similarity
    sorted_images = dict(sorted(my_images.items(), key=operator.itemgetter(1)))

    # finds the minimum distance of the sorted values for overall matching score
    minimum = min(sorted_images.values())

    # finds the maximum distance of the sorted values for overall matching score
    maximum = max(sorted_images.values())

    # intializing dictionary for the top "k" similar images
    top_images={}

    #calculates a similarity score and stores top "k" in the dictionary with image_id
    for h, i in enumerate(sorted_images):
        if (h < k):
            top_images.update({i:sorted_images[i]})

            # match score is calculated by taking 1 - the current distance and dividing it by the range of distances
            match_score = 1 - (((sorted_images[i])/(maximum - minimum)))
            
            # formats the score as a percentage
            match_score = "{:.0%}".format(match_score)
            top_images[i] = match_score
    
    # header title for the task 3 output
    print("Top " + str(k) + " Most Similar Images: ")

    # generates output on the command line for top image scores with their image_id
    for image, score in top_images.items():
        print("{} ({})".format(image, score))

# this main program runs the interactive GUI for user input on command line
def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    print("Welcome to Phase 1!")
    while True:
        print("---------------------------------------------------------------")
        start = input("Please press enter/return to continue or enter q to quit: ")
        if (start == "q"):
            break
        task = input("Which task number would you like to run: [1] [2] [3]? ")
        if (task == "1"):
            image_id = input("Please enter the full path of the image: ")
            feature_descriptor = input("Which feature descriptor would you like to select: [lbp] [hog]? ")
            print("************** Task 1 Running *****************")
            task1(image_id, feature_descriptor)
            print("************** Task 1 Complete *****************")
        elif (task == "2"):
            directory = input("Please enter the full path of the image directory: ")
            feature_descriptor = input("Which feature descriptor would you like to select [lbp] [hog]? ")
            print("************** Task 2 Running *****************")
            task2(directory, feature_descriptor)
            print("************** Task 2 Complete *****************")
        elif (task == "3"):
            image_id = input("Please enter the full path of the image: ")
            feature_descriptor = input("Which feature descriptor would you like to select [lbp] [hog]? ")
            k = input("Please enter a number for how many similar images you would like to match: ")
            print("************** Task 3 Running *****************")
            task3(image_id, feature_descriptor, k)
            print("************** Task 3 Complete *****************")

# implements main method
if __name__== "__main__":
  main()

