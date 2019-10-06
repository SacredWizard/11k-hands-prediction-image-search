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
import numpy as np
import time
from classes.mongo import MongoWrapper
from classes.global_constants import GlobalConstants
from sklearn.decomposition import NMF
from scipy.linalg import svd

class DimensionReduction:
    """
    Class for performing Dimensionality Reduction
    """
    def __init__(self, extractor_model, dimension_reduction_model, k_value):
        self.constants = GlobalConstants()
        self.mongo_wrapper = MongoWrapper(self.constants.Mongo().DB_NAME)
        self.extractor_model = extractor_model
        self.dimension_reduction_model = dimension_reduction_model
        self.k_value = k_value
        pass

    def get_object_feature_matrix(self):
        vector_list = []
        cursor = self.mongo_wrapper.find(self.extractor_model.lower(), {}, {'_id': 0})
        if self.extractor_model == 'LBP':
            for rec in cursor:
                vector_list.append(rec['featureVector'].split(','))
            return np.array(vector_list).astype(np.float)
        else:
            for rec in cursor:
                vector_list.append(rec['featureVector'])
            return np.array(vector_list)

    def execute(self):
        """Performs dimensionality reduction"""
        return getattr(DimensionReduction, self.dimension_reduction_model)(self)

    def pca(self):
        pass

    def svd(self):
        data = self.get_object_feature_matrix()
        if data:
            # Singular-value decomposition
            U, s, VT = svd(feature_matrix)

            # gets absolute value
            newVT = abs(VT[:k, :])
            newU = abs(U[:,:k])

            return newU, newVT

    def nmf(self):
        data = self.get_object_feature_matrix()
        if data:
            model = NMF(n_components=self.k_value, beta_loss='frobenius', init='nndsvd', random_state=0)
            print(model)
            w = model.fit_transform(data)
            h = model.components_
            tt1 = time.time()
            for i in range(h.shape[0]):
                print("Latent Feature: {}\n{}".format(i + 1, sorted(((i, v) for i, v in enumerate(H[i])),
                                                                    key=lambda x: x[1], reverse=True)))

            print("Time NMF {}".format(time.time() - tt1))
            return w

    def lda(self):
        pass
