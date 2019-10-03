#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:55:44 2019

@author: karishmajoseph, baani-khurana
"""

from numpy import array
from numpy import diag
from numpy import zeros
from numpy import argsort
import pymongo
import numpy as np

from scipy.linalg import svd
from classes import global_connections, global_constants
from classes.featureextraction import ExtractFeatures

connection = global_connections.GlobalConnections()

def implement_svd(feature_matrix, k, feature_descriptor):

    # Singular-value decomposition
    U, s, VT = svd(feature_matrix)

    # gets absolute value
    newVT = abs(VT[:k, :])
    newU = abs(U[:,:k])

    # creates the term weight pairs
    U_TW = getDataLatentSemantics(newU, k)
    VT_TW = getFeatureLatentSemantics(newVT, k)

    # prints results to output file
    saveTW(U_TW, VT_TW, feature_descriptor)

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

#saves Term Weight Pairs to txt files
def saveTW(U_TW, VT_TW, feature_descriptor):
    f = open(feature_descriptor + "_svd.txt", "w+")
    f.write("Data Latent semantics\n")
    f.write(str(U_TW) + "\n")
    f.write("Feature Latent semantics\n")
    f.write(str(VT_TW) + "\n")
    f.close()

def main():
    # define a matrixet_object_featur
    feature_descriptor = 'CM'
    feature_matrix = get_object_feature_matrix(feature_descriptor)
    k = 3
    if (k > feature_matrix.shape[1]):
        print("This k value is invalid. Please enter a different k value.")
    else:
        implement_svd(feature_matrix, k, feature_descriptor)
        print("Done")

def get_object_feature_matrix(extractor):
    try:
        vector_list = []
        cursor = connection.mongo_client.features[extractor.lower()].find({}, {'_id': 0})
        if extractor == 'LBP':
            for rec in cursor:
                vector_list.append(rec['featureVector'].split(','))
            return np.array(vector_list).astype(np.float)

        for rec in cursor:
            vector_list.append(rec['featureVector'])
        return np.array(vector_list)

    except pymongo.errors.ServerSelectionTimeoutError as e:
        print("Timeout:\n{}".format(e))
    except Exception as e:
        print("Exception occurred:\n{}".format(e))


if __name__ == "__main__":
    main()
