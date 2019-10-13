#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:55:44 2019

@author: karishmajoseph, baani-khurana
"""

import numpy as np
import pymongo
from numpy import argsort
from scipy.linalg import svd

from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper


def implement_svd(feature_matrix, k, feature_descriptor):

    # Singular-value decomposition
    U, s, VT = svd(feature_matrix)

    # gets absolute value
    newVT = abs(VT[:k, :])
    newU = abs(U[:,:k])

    # creates the term weight pairs
    U_TW = getDataLatentSemantics(newU, k)
    VT_TW = getFeatureLatentSemantics(newVT, k)

    # prints results to console
    printTW(U_TW, VT_TW, feature_descriptor)

    return newU, newVT

def getDataLatentSemantics(U_matrix, k):
    term = [argsort(-U_matrix[:, i]) for i in range(k)]
    TW=[]
    #for each latent semantic (k)
    for i in range(k):
        kTW = []
        #for each data object
        for j in range(len(term[i])):
            kTW.append([term[i][j],U_matrix[term[i][j],i]])
        TW.append(kTW)
    return TW

def getFeatureLatentSemantics(VT_matrix, k):
    term = [argsort(-VT_matrix[i, :]) for i in range(k)]
    TW=[]
    #for each latent semantic (k)
    for i in range(k):
        kTW = []
        #for each feature
        for j in range(len(term[i])):
            kTW.append([term[i][j],VT_matrix[i,term[i][j]]])
        TW.append(kTW)
    return TW

#prints Term Weight Pairs to console
def printTW(U_TW, VT_TW, feature_descriptor):
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Data Latent semantics\n")
    print(U_TW)
    print()
    print("-------------------------------------------------------------------------------------------------------------")
    print()
    print("Feature Latent semantics\n")
    print(VT_TW)

def main():
    # define a matrixet_object_featur
    feature_descriptor = 'CM'
    feature_matrix = get_object_feature_matrix(feature_descriptor)
    k = 3
    print(feature_matrix)
    if (k > feature_matrix.shape[1]):
        print("This k value is invalid. Please enter a different k value.")
    else:
        U, VT = implement_svd(feature_matrix, k, feature_descriptor)
        print("Done")

def get_object_feature_matrix(extractor_model):
    constants = GlobalConstants()
    mongo_wrapper = MongoWrapper(constants.Mongo().DB_NAME)
    try:
        vector_list = []
        cursor = mongo_wrapper.find(extractor_model.lower(), {}, {'_id': 0})
        if extractor_model == 'LBP':
            for rec in cursor:
                vector_list.append(rec['featureVector'].split(','))
            return np.array(vector_list).astype(np.float)
        else:
            for rec in cursor:
                vector_list.append(rec['featureVector'])
            return np.array(vector_list)

    except pymongo.errors.ServerSelectionTimeoutError as e:
        print("Timeout:\n{}".format(e))
    except Exception as e:
        print("Exception occurred:\n{}".format(e))


if __name__ == "__main__":
    main()
