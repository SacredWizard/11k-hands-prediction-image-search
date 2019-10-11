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

import glob
import os
from multiprocessing.pool import ThreadPool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage.feature import hog, local_binary_pattern
from skimage.transform import rescale

import utils.validate as validate
from classes import global_constants
from classes import mongo


class ExtractFeatures:
    """
    Class for Extracting features from images.
    Initialize the class by passing the folder, model and image(optional).
    And use the method 'execute' to extract features.
    If both folder and image are passed, the features are calculated for the given image and model and returned.
    If just the folder is passed, the features are computed for all images in the folder for the given model and stored
    in mongo.
    """

    def __init__(self, folder, model):
        """Init function for the Feature Extraction class"""
        if not validate.validate_folder(folder):
            raise Exception('Input Parameters are incorrect, Pass Valid Folder and Model Name')
        self.constants = global_constants.GlobalConstants()
        self.folder = folder
        self.model = model
        # if image and image.endswith(self.constants.JPG_EXTENSION):
        #     try:
        #         if not Image.open(os.path.join(self.folder, image)).format == 'JPEG':
        #             raise Exception('Not a Valid JPEG Image, Pass a correct file')
        #     except IOError as e:
        #         raise Exception('Error in Opening File:\n{}'.format(e))
        # self.image = image
        # self.feature = None

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

    def execute(self, image=None):
        """Extract Features from Images"""
        if image:
            # return globals()["extract_" + self.model.lower()]()
            return getattr(ExtractFeatures, "extract_{}".format(self.model.lower()))(self, image)
        else:
            self.extract_features_folder()

    def extract_cm(self, image_name, process_record=False, window_size=100):
        """
        Calculate Color Moments for an Image (Image is converted to yuv format)
        Three color moments are computed namely, Mean, Standard Deviation, Skew
        Image is divided into windows specified by a window size and then Color Moments are computed
        Size of a block = window_size * window_size
        :param process_record: Convert the ouput in a form to store in the database (JSON Format)
        :param image_name: Image ID (An Identifier for an Image)
        :param window_size: The Window size of the image used to split the image into blocks having a size of window
        :return: Computed Color Moments
        """
        if not validate.validate_image(self.folder, image_name):
            raise Exception("Image filename doesnot exist")
        # image_id = self.image
        img = cv2.imread(os.path.join(self.folder, image_name))
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

        if process_record:
            return {'imageId': image_name, 'featureVector': y + u + v}
        return y + u + v

    def extract_hog(self, image_name, process_record=False, show=False):
        """
        Extracts Histogram of gradients
        :param image_name:
        :param process_record:
        :param show:
        :return:
        """

        if not validate.validate_image(self.folder, image_name):
            raise Exception('File is not valid')

        image = io.imread(os.path.join(self.folder, image_name))
        rescale_image = rescale(image, 0.1, anti_aliasing=True)

        fd, hog_image = hog(rescale_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            visualize=True, block_norm='L2', multichannel=True, feature_vector=True)
        if show:
            plt.imshow(hog_image)
            plt.show()
        if process_record:
            return {'imageId': image_name, 'featureVector': fd.ravel().tolist()}
        return fd.ravel().tolist()

    def extract_lbp(self, image_name, process_record=False):
        """
        Method to extract LBP
        :return: None, sets the class variable
        """

        if not validate.validate_image(self.folder, image_name):
            raise Exception('File is not valid')

        # parameter settings for local binary pattern calculation
        lbp_constants = self.constants.Lbp()
        image = color.rgb2gray(io.imread(os.path.join(self.folder, image_name)))
        lbp_list = np.asarray([])

        # splits the 1600x1200 image into 192 blocks of 100x100 pixels and calculates lbp vector for each block
        for row in range(0, image.shape[0], self.constants.WINDOW_SIZE):
            for column in range(0, image.shape[1], self.constants.WINDOW_SIZE):
                window = image[row:row + self.constants.WINDOW_SIZE, column:column + self.constants.WINDOW_SIZE]
                lbp = local_binary_pattern(
                    window, lbp_constants.NUM_POINTS, lbp_constants.RADIUS, lbp_constants.METHOD_UNIFORM)
                window_histogram = np.histogram(lbp, bins=lbp_constants.BINS)[0]
                lbp_list = np.append(lbp_list, window_histogram)
        lbp_list.ravel()
        if process_record:
            return {'imageId': image_name, 'featureVector': lbp_list.tolist()}
        return lbp_list.tolist()

    def extract_sift(self, image_name, process_record=False):
        """
        Calculate SIFT Features for an image
        :param process_record:
        :param image_name:
        :param image: Image name (File Name of the image)
        :return:
        """

        if not validate.validate_image(self.folder, image_name):
            raise Exception('File is not valid')

        # image_id = image
        img = cv2.imread(os.path.join(self.folder, image_name))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sft = cv2.xfeatures2d.SIFT_create()
        kp, des = sft.detectAndCompute(gray_img, None)
        # sift_features = [{"keyPoint": i, "x": kp[i].pt[0], "y": kp[i].pt[1], "angle": kp[i].angle,
        #                   "size": kp[i].size, "descriptor": des[i].tolist()} for i in range(len(kp))]
        # return {"imageId": image_id, "kpCount": len(kp), "features": sift_features}
        if process_record:
            return {'imageId': image_name, 'featureVector': des}
        return des

    def extract_features_folder(self):
        """
        Method for extracting features for all images in a folder
        :return:
        """
        file_names = sorted(glob.glob1(self.folder, '*' + self.constants.JPG_EXTENSION))
        # try:
        length = len(file_names)
        mongo_wrapper = mongo.MongoWrapper()
        for i in range(0, length, self.constants.BULK_PROCESS_COUNT):
            pool = ThreadPool(self.constants.NUM_THREADS)
            mongo_wrapper.bulk_insert(self.model.lower(), pool.starmap(
                getattr(
                    ExtractFeatures, 'extract_' + self.model.lower()), [(self, i, True) for i in file_names[i: length]]
                if i + self.constants.BULK_PROCESS_COUNT > length
                else [(self, i) for i in file_names[i: i + self.constants.BULK_PROCESS_COUNT]]))

            # mongo_wrapper.bulk_insert(self.model.lower(), pool.starmap(
            #     getattr(ExtractFeatures, 'extract_' + self.model.lower())(self, [i for i in file_names[i: length]]
            #     if i + self.constants.BULK_PROCESS_COUNT > length
            #     else [i for i in file_names[i: i + self.constants.BULK_PROCESS_COUNT]])))

        print("Successfully Inserted Feature Vectors for {} images".format(length))
        # except Exception as e:
        #     print('Error:\n{}'.format(e))
