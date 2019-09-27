"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is a module for performing feature extraction on images
"""

from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure, io, color
from skimage.transform import downscale_local_mean, rescale
import os
import numpy
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys


class ExtractFeatures:
    """
    Class for Extracting features from images.
    Initialize the class by passing the folder, model and image(optional).
    And use the method 'execute' to extract features.
    If both folder and image are passed, the features are calculated for the given image and model and returned.
    If just the folder is passed, the features are computed for all images in the folder for the given model and stored
    in mongo.
    """

    def __init__(self, folder, model, image=None):
        """Init function for the Feature Extraction class"""
        self.folder = folder
        self.model = model
        self.allowed_models = ["HOG", "SIFT", "CM", "LBP"]
        if self.model not in self.allowed_models:
            print("Invalid Model passed")
            sys.exit(1)
        self.image = image
        self.feature = None

    def execute(self):
        """Extract Features from Images"""
        if self.image:
            if self.model == "HOG":
                self.extract_hog()
            elif self.model == "CM":
                self.extract_cm()
            elif self.model == "LBP":
                self.extract_lbp()
            elif self.model == "SIFT":
                self.extract_sift()
            return self.feature
        else:
            self.extract_features_folder()

    def extract_lbp(self):
        """Method for extracting local binary patterns"""
        # parameter settings for local binary pattern calculation
        num_points = 8
        radius = 1

        # name of the file is the image_id pathway
        filename = os.path.join(self.folder, self.image)

        # reads in the image
        image = io.imread(filename)

        # converts image to grayscale
        image = color.rgb2gray(image)

        # dimensions for 100x100 windows for the image
        window_rows = 100
        window_columns = 100

        lbp_list = []

        # splits the 1600x1200 image into 192 blocks of 100x100 pixels and calculates lbp vector for each block
        for row in range(0, image.shape[0], window_rows):
            for column in range(0, image.shape[1], window_columns):
                window = image[row:row + window_rows, column:column + window_columns]
                lbp = local_binary_pattern(window, num_points, radius, 'uniform')
                # appends the lbp vector for each window into a concatenated list
                lbp_list.append(lbp.tolist())

        lbp_list = numpy.asarray(lbp_list).ravel()
        self.feature = str(lbp_list.tolist())

    def extract_sift(self):
        pass

    def extract_hog(self, show=True):
        """Method for extracting histogram of oriented gradients"""
        image = io.imread(os.path.join(self.folder, self.image))
        down_image = downscale_local_mean(image, (10, 10, 1))
        rescale_image = rescale(image, 0.1, anti_aliasing=True)
        down_image = down_image.astype(numpy.uint8)

        fd, hog_image = hog(rescale_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, block_norm='L2', multichannel=True,
                            feature_vector=True)
        if show:
            plt.imshow(hog_image)
            plt.show()

        self.feature = fd.ravel().tolist()

    def extract_features_folder(self):
        """Method for extracting features for all images in a folder"""
        formats = [".jpeg", ".png", ".jpg"]
        image_list = [file for file in os.listdir(self.folder)
                      if (os.path.isfile(os.path.join(self.folder, file)) and file.endswith(tuple(formats)))]
        image_list = sorted(image_list)

        for img in image_list:
            self.image = img
            if self.model == "HOG":
                self.extract_hog(show=False)
            elif self.model == "CM":
                self.extract_cm()
            elif self.model == "SIFT":
                self.extract_sift()
            elif self.model == "LBP":
                self.extract_lbp()

            self.save_feature_mongo()
        print("Successfully Inserted Feature Vectors for {} images".format(len(image_list)))

    def extract_cm(self):
        """Method for extracting color moments"""
        features = [self.image]
        return features

    def save_feature_mongo(self):
        """Method for saving extracted features to mongo"""
        mongo_client = MongoClient()
        try:
            mongo_client.features[self.model.lower()].update({"ImageId": "{}".format(self.image)},
                                                             {"$set": {"ImageId": "{}".format(self.image),
                                                                       "featureVector": self.feature}}, upsert=True)
            print("Successfully Inserted the feature vector for {}".format(self.image))
        except Exception as exp:
            print("Error while inserting the record into Mongo:{}".format(exp))
            sys.exit(2)
