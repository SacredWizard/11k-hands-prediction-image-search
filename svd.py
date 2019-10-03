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
from scipy.linalg import svd

from classes.featureextraction import ExtractFeatures

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
    #for each latent semantic
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
    #for each latent semantic
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
    # define a matrix
    feature_matrix = array([
        [1,1,1,0,0],
        [3,3,3,0,0],
        [4,4,4,0,0],
        [5,5,5,0,0],
        [0,2,0,4,4],
        [0,0,0,5,5],
        [0,1,0,2,2]])
    k = 6
    if (k > feature_matrix.shape[1]):
        print("This k value is invalid. Please enter a different k value.")
    else:
        implement_svd(feature_matrix, k, "hog")
        print("Done")

def createFeatureMatrix(feature_descriptor):
    array = []


if __name__ == "__main__":
    main()
