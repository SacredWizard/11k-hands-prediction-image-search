"""
Multimedia Web Databases - Fall 2019: Project Group 17
Authors:
1. Sumukh Ashwin Kamath
2. Rakesh Ramesh
3. Baani Khurana
4. Karishma Joseph
5. Shantanu Gupta
6. Kanishk Bashyam

This is a module for performing dimensionality reduction on images
"""
import re
import time
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD, PCA
from sklearn.preprocessing import normalize

import utils.distancemeasure
from classes.featureextraction import ExtractFeatures
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper


class DimensionReduction:
    """
    Class for performing Dimensionality Reduction
    """
    def __init__(self, extractor_model, dimension_reduction_model, k_value, label=None):
        self.constants = GlobalConstants()
        self.mongo_wrapper = MongoWrapper(self.constants.Mongo().DB_NAME)
        self.extractor_model = extractor_model
        self.dimension_reduction_model = dimension_reduction_model
        self.label = label
        self.k_value = k_value

    def get_object_feature_matrix(self):
        """
        Returns The Object feature Matrix
        :param mapping: Default: False, if mapping is True, Returns the object feature matrix with image mappings
        :return: The Object Feature Matrix
        """
        cursor = self.mongo_wrapper.find(self.extractor_model.lower(), {"path": {"$exists": True}}, {'_id': 0})
        if cursor.count() > 0:
            df = pd.DataFrame(list(cursor))

            if self.label:
                filter_images_list = self.filter_images_by_label(df['imageId'].tolist())
                df = df[df.imageId.isin(filter_images_list)]
            return df
        else:
            return pd.DataFrame()

    def filter_images_by_label(self, images_list):
        """Fetches the list of images by label"""
        query = {"imageName": {"$in": images_list}}

        if self.label == "left-hand" or self.label == "right-hand" or self.label == "dorsal" or self.label == "palmar":
            query['aspectOfHand'] = {"$regex": re.sub('-hand$', '', self.label)}
        elif self.label == "male" or self.label == "female":
            query['gender'] = self.label
        elif self.label == "with accessories":
            query['accessories'] = 1
        elif self.label == "without accessories":
            query['accessories'] = 0
        else:
            raise Exception("Incorrect Label")

        filter_images_list = [d['imageName'] for d in list(self.mongo_wrapper.find(
            self.constants.METADATA, query, {"imageName": 1, "_id": 0}))]

        return filter_images_list

    def execute(self):
        """Performs dimensionality reduction"""
        return getattr(DimensionReduction, self.dimension_reduction_model.lower())(self)

    # method to perform Principal Component Analysis on n-dimensional features 
    def pca(self):
        data = self.get_object_feature_matrix()
        # get object-feature vectors matrix
        data_feature_matrix = np.array(data['featureVector'].tolist())
        k = self.k_value
        if not data_feature_matrix.size == 0:
            # normalize feature vector data for PCA
            normalize(data_feature_matrix)
            # apply PCA to features
            features_pca_decomposition = PCA(n_components=k,copy=False)
            features_pca_decomposition.fit_transform(data_feature_matrix)
            # get latent feature components
            feature_components = features_pca_decomposition.components_
            
            data_pca_decomposition = PCA(n_components=k,copy=False)
            # transpose matrix to feature-data matrix
            feature_data_matrix = np.transpose(data_feature_matrix)
            # normalize feature vector data for PCA
            normalize(feature_data_matrix)
            # apply PCA to features
            fit = data_pca_decomposition.fit_transform(feature_data_matrix)
            # get latent data components
            data_components = np.transpose(data_pca_decomposition.components_)
            # map imageID with principal components
            img_dim_mapping = pd.DataFrame({"imageId": data['imageId'], "reducedDimensions": data_components.tolist()})
            return img_dim_mapping, feature_components, features_pca_decomposition
        raise \
            Exception("Data is empty in database, run Task 2 of Phase 1 (Insert feature extracted records in db )\n\n")
        # 
        

    def svd(self):
        data = self.get_object_feature_matrix()
        obj_feature = np.array(data['featureVector'].tolist())

        k = self.k_value
        if obj_feature is not None:
            # Singular-value decomposition
            svd_model = TruncatedSVD(n_components=k)
            U = svd_model.fit_transform(obj_feature)
            U = pd.DataFrame({"imageId": data['imageId'], "reducedDimensions": U.tolist()})
            VT = svd_model.components_

            return U, VT, svd_model

    def nmf(self):
        """
        Performs NMF dimensionality reduction
        :return:
        """
        constants = self.constants.Nmf()
        data = self.get_object_feature_matrix()

        if not data.size == 0:
            obj_feature = np.array(data['featureVector'].tolist())
            if (obj_feature < 0).any():
                print("NMF does not accept negative values")
                return

            model = NMF(n_components=self.k_value, beta_loss=constants.BETA_LOSS_FROB
                        , init=constants.INIT_MATRIX, random_state=0)
            w = model.fit_transform(obj_feature)
            h = model.components_
            tt1 = time.time()
            data_lat = pd.DataFrame({"imageId": data['imageId'], "reducedDimensions": w.tolist()})
            for i in range(h.shape[0]):
                print("Latent Feature: {}\n{}".format(i + 1, sorted(((i, v) for i, v in enumerate(h[i])),
                                                                    key=lambda x: x[1], reverse=True)))

            print("\n\nTime Taken for NMF {}\n".format(time.time() - tt1))
            return data_lat, h, model
        raise \
            Exception("Data in database is empty, Run Task 2 of Phase 1 (Insert feature extracted records in db )\n\n")

    def lda(self):
        """
        Performs LDA Dimensionality reduction
        :return:
        """
        data = self.get_object_feature_matrix()
        obj_feature = np.array(data['featureVector'].tolist())

        if (obj_feature < 0).any():
            print("LDA does not accept negative values")
            return

        model = LatentDirichletAllocation(n_components=self.k_value, max_iter=40, random_state=0, learning_decay=.75,
                                          learning_method='online')
        # topic_word_prior=0.05, doc_topic_prior=0.01)#learning_method='online')
        lda_transformed = model.fit_transform(obj_feature)
        data_lat = pd.DataFrame({"imageId": data['imageId'], "reducedDimensions": lda_transformed.tolist()})

        # Compute model_component in terms of probabilities
        model_comp = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

        return data_lat, model.components_, model

    def compute_query_image(self, model, folder, image):
        """
        Computes the reduced dimensions for the new query image
        :param model: Learned model
        :param folder: Folder in which the query image is
        :param image: Filename of the query image
        :return: Reduced Dimensions for the new vector
        """
        feature_extractor = ExtractFeatures(folder, self.extractor_model)
        result = feature_extractor.execute(image)
        return model.transform([result])

    def find_m_similar_images(self, model, m, folder, image, dist_func):
        """
        Finds m similar images to the given query image
        :param m: The integer value of m
        :param model: The learned model which is saved
        :param folder: Folder in which the given query image is present
        :param image: Filename of the query image
        :param dist_func: Distance function to be used
        :return: m similar images with their scores
        """
        query_reduced_dim = self.compute_query_image(model, folder, image)
        obj_feature = self.get_object_feature_matrix()
        dist = []
        score = []
        for index, row in obj_feature.iterrows():
            dist.append(getattr(utils.distancemeasure, dist_func)(query_reduced_dim,
                                                                  model.transform([row['featureVector']])))
        for d in dist:
            if dist_func == "nvsc1":
                score.append(d * 100)
            else:
                score.append((1 - d/max(dist)) * 100)

        obj_feature['dist'] = dist
        obj_feature['score'] = score

        obj_feature = obj_feature.sort_values(by="score", ascending=False)

        result = []
        for index, row in islice(obj_feature.iterrows(), m):
            rec = dict()
            rec['imageId'] = row['imageId']
            rec['score'] = row['score']
            rec['path'] = row['path']
            result.append(rec)
        return result
