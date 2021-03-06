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
from sklearn.cluster import MiniBatchKMeans

import utils.validate as validate
from classes import globalconstants
from classes import mongo
from utils.model import Model


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
        self.constants = globalconstants.GlobalConstants()
        self.folder = folder.rstrip("/")
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
            return getattr(ExtractFeatures, "extract_{}".format(self.model.lower()))(self, image)
        else:
            self.extract_features_folder()

    def extract_cm(self, image_name, process_record=False, window_size=100):
        """
        Calculate Color Moments for an Image (Image is converted to yuv format)
        Three color moments are computed namely, Mean, Standard Deviation, Skew
        Image is divided into windows specified by a window size and then Color Moments are computed
        Size of a block = window_size * window_size
        :param process_record: Convert the output in a form to store in the database (JSON Format)
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
            return {'imageId': image_name, 'featureVector': y + u + v,
                    "path": os.path.abspath(os.path.join(self.folder, image_name))}
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
            return {'imageId': image_name, 'featureVector': fd.ravel().tolist(),
                    "path": os.path.abspath(os.path.join(self.folder, image_name))}
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
            return {'imageId': image_name, 'featureVector': lbp_list.tolist(),
                    "path": os.path.abspath(os.path.join(self.folder, image_name))}
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

        img = cv2.imread(os.path.join(self.folder, image_name))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sft = cv2.xfeatures2d.SIFT_create()
        kp, des = sft.detectAndCompute(gray_img, None)
        if process_record:
            return {'imageId': image_name,
                    'kps': [{'x': k.pt[0], 'y': k.pt[1], 'size': k.size, 'angle': k.angle, 'response': k.response}
                            for k in kp], 'featureVector': [i.tolist() for i in des],
                    "path": os.path.abspath(os.path.join(self.folder, image_name))}
        model_file = "{}_{}_{}".format(self.folder, self.model.lower(), self.constants.BOW_MODEL.lower())

        if validate.validate_file(os.path.join(self.constants.MODELS_FOLDER, model_file)):
            model = Model()
            knn = model.load_model(model_file)
            histogram = np.zeros(knn.n_clusters)
            for desc in des:
                index = knn.predict([desc])
                histogram[index] += 1 / len(kp)
            return histogram
        return des

    def create_bog_histogram(self, overwrite=False):
        """
        Convert Sift Features into Bag of Words Representation
        :return:
        """
        mongo_wrapper = mongo.MongoWrapper()
        cursor = mongo_wrapper.find(self.model.lower() + '_features', {}, {'_id': 0, 'featureVector': 1, 'imageId': 1})
        model_file_name = "{}_{}_{}".format(self.folder, self.model.lower(), self.constants.BOW_MODEL.lower())
        if not validate.validate_file(os.path.join(self.constants.MODELS_FOLDER, model_file_name)) or overwrite:
            model = Model()
            max_kp_count = 0
            feature_data = {}
            feature_data_list = []
            for rec in cursor:
                length = len(rec['featureVector'])
                max_kp_count = length if length > max_kp_count else max_kp_count
                feature_data[rec['imageId']] = rec['featureVector']
            descriptor_values = feature_data.values()
            knn = MiniBatchKMeans(init='random',
                init_size=5 * max_kp_count, n_clusters=max_kp_count, batch_size=self.constants.BOW_BATCH_SIZE). \
                fit(np.array([item for descriptor_values in descriptor_values for item in descriptor_values]))
            model.save_model(knn, model_file_name)
            histogram_list = []
            for key in feature_data.keys():
                descriptors = feature_data[key]
                desc_count = len(descriptors)
                histogram = np.zeros(max_kp_count)
                for descriptor in descriptors:
                    index = knn.predict([descriptor])
                    histogram[index] += 1 / desc_count
                histogram_list.append(histogram)
                feature_data_list.append({'imageId': key, 'featureVector': histogram.tolist(),
                                          "path": os.path.abspath(os.path.join(self.folder, key))})
            model.save_model(np.asarray(histogram_list), "{}_{}".format(model_file_name, 'bow_histogram'))
            mongo_wrapper.bulk_insert(self.model.lower(), feature_data_list)

    def extract_features_folder(self):
        """
        Method for extracting features for all images in a folder
        :return:
        """
        file_names = sorted(glob.glob1(self.folder, '*' + self.constants.JPG_EXTENSION))
        length = len(file_names)
        mongo_wrapper = mongo.MongoWrapper()
        # Dropping the collection before bulk inserting
        mongo_wrapper.drop_collection(self.model.lower())
        if self.model == self.constants.SIFT:
            mongo_wrapper.drop_collection(mongo_wrapper.constants.SIFT_FEATURE_COLLECTION.lower())

        for i in range(0, length, self.constants.BULK_PROCESS_COUNT):
            pool = ThreadPool(self.constants.NUM_THREADS)
            mongo_wrapper.bulk_insert(self.model.lower() if self.model != self.constants.SIFT
                                      else mongo_wrapper.constants.SIFT_FEATURE_COLLECTION, pool.starmap(
                getattr(
                    ExtractFeatures, 'extract_' + self.model.lower()), [(self, i, True) for i in file_names[i: length]]
                if i + self.constants.BULK_PROCESS_COUNT > length
                else [(self, i, True) for i in file_names[i: i + self.constants.BULK_PROCESS_COUNT]]))
        if self.model == self.constants.SIFT:
            print('Processing Data for {}'.format(self.model))
            self.create_bog_histogram(overwrite=True)

            # mongo_wrapper.bulk_insert(self.model.lower(), pool.starmap(
            #     getattr(ExtractFeatures, 'extract_' + self.model.lower())(self, [i for i in file_names[i: length]]
            #     if i + self.constants.BULK_PROCESS_COUNT > length
            #     else [i for i in file_names[i: i + self.constants.BULK_PROCESS_COUNT]])))
