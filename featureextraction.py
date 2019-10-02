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

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from skimage import io, color
from skimage.feature import hog, local_binary_pattern
from skimage.transform import downscale_local_mean, rescale


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

        lbp_list = np.asarray([])

        # splits the 1600x1200 image into 192 blocks of 100x100 pixels and calculates lbp vector for each block
        for row in range(0, image.shape[0], window_rows):
            for column in range(0, image.shape[1], window_columns):
                window = image[row:row + window_rows, column:column + window_columns]
                lbp = local_binary_pattern(window, num_points, radius, 'uniform')
                lbp_list = np.append(lbp_list, lbp)
        lbp_list.ravel()
        self.feature = str(lbp_list.tolist()).strip('[]')

    def extract_sift(self):
        """
        Calculate SIFT Features for an image
        :param img: Image in BGR Format (Default format used to read Image in OpenCV)
        :param image_id: Image ID (An Identifier for an Image)
        :return:
        """
        image_id = self.image
        img = cv2.imread(os.path.join(self.folder, self.image))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sft = cv2.xfeatures2d.SIFT_create()
        kp, des = sft.detectAndCompute(gray_img, None)
        sift_features = [{"keyPoint": i, "x": kp[i].pt[0], "y": kp[i].pt[1], "angle": kp[i].angle,
                          "size": kp[i].size, "descriptor": des[i].tolist()} for i in range(len(kp))]
        self.feature = {"imageId": image_id, "kpCount": len(kp), "features": sift_features}

    def extract_hog(self, show=True):
        """Method for extracting histogram of oriented gradients"""
        image = io.imread(os.path.join(self.folder, self.image))
        down_image = downscale_local_mean(image, (10, 10, 1))
        rescale_image = rescale(image, 0.1, anti_aliasing=True)
        down_image = down_image.astype(np.uint8)

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

    def extract_cm(self, window_size=100):
        """
        Calculate Color Moments for an Image (Image is converted to yuv format)
        Three color moments are computed namely, Mean, Standard Deviation, Skew
        Image is divided into windows specified by a window size and then Color Moments are computed
        Size of a block = window_size * window_size
        :param img: Image in BGR Format (Default format used to read Image in OpenCV)
        :param image_id: Image ID (An Identifier for an Image)
        :param window_size: The Window size of the image used to split the image into blocks having a size of window
        :return: Computed Color Moments
        """
        # image_id = self.image
        img = cv2.imread(os.path.join(self.folder, self.image))
        img_yuv = self.bgr2yuv(img)
        (rows, cols) = img.shape[:2]
        # data = {"imageId": image_id, "type": "CM"}
        counter = 0
        y, u, v = [], [], []

        for i in range(0, int(rows / window_size)):
            for j in range(0, int(cols / window_size)):
                (mean, std_dev) = cv2.meanStdDev(
                    img_yuv[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size])
                skw = self.skew(img_yuv[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size])
                counter += 1
                y = y + [mean[0][0], std_dev[0][0], skw[0]]
                u = u + [mean[1][0], std_dev[1][0], skw[1]]
                v = v + [mean[2][0], std_dev[2][0], skw[2]]
        self.feature = y + u + v

    def skew(self, img):
        """
        Calculate Skew for an image
        :param img: Image
        :return: Skew for individual image channels
        """
        (rows, cols, colors) = img.shape
        mean = [np.mean(img[:, :, i]) for i in range(colors)]
        return \
            [(np.cbrt(np.divide(np.sum(np.power(np.subtract(img[:, :, i], mean[i]), 3)), rows * cols))) for i in
             range(colors)]

    def bgr2yuv(self, img):
        """
        Convert a BGR Image to YUV Image
        :param img: BGR Image
        :return: YUV Image
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

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
