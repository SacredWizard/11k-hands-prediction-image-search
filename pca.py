<<<<<<< HEAD
=======
import math
import sys
import time
import argparse
import numpy
import sys
from featureextraction import ExtractFeatures
import os
import warnings

import os
import sys

import cv2
import operator
import matplotlib.pyplot as plt
>>>>>>> commit intermediate code for pca
import numpy as np
from pymongo import MongoClient
<<<<<<< HEAD
=======
from skimage import io, color
from skimage.feature import hog, local_binary_pattern
from skimage.transform import downscale_local_mean, rescale
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import csv
import time
warnings.filterwarnings("ignore")

def pca(arr,_k):
    start_time = time.clock()

    # Normalize data-feature vector array along column
    meanNormalized = arr
    meanNormalized = normalize(arr,axis=0,copy=False)
    
    # Again normalize feature vector array along row
    meanNormalized = normalize(meanNormalized,axis=1,copy=False)
    # Calculate the covariance matrix, rowvar=False means calculate covariance of feature vectors 
    covMatrix = np.cov(meanNormalized, rowvar=False)

    # Calculate eigen values and eigen vectors of the covariane matrix of feature vectors
    eigVal,eigVec = np.linalg.eigh(covMatrix)

    # print("eigVec\n",eigVec)
    kEigVectors = eigVec[:,:_k]
    # print("kEigVectors\n",kEigVectors)
    # Get the index of original eigen values in ascending order
    # sorted = eigVal.argsort()[::-1][:_k]
    # sortedEigenVectors = eigVec.argsort()[::-1]
    outputArray = []
    with open("C://topLatentSemantics.csv", mode='w', newline='') as csvfile:
        _csvWriter = csv.writer(csvfile, delimiter=",")
        # for i in sorted:
        #     _csvWriter.writerow([i,eigVal[i]])
        for i in range (0,np.shape(kEigVectors)[1]):
            termIndexes = (kEigVectors[:,i]).argsort()[::-1]
            # print(i," termIndexes ",termIndexes)
            keyValue = []
            for index,j in enumerate(termIndexes):
                keyValue.append(tuple((j, kEigVectors[j][i])))
            _csvWriter.writerow(["PC"+str(i)+" : ",keyValue])

    # sort the eigen values and eigen vectors based on increasing eigen values
    # eigVal,eigVec = eigVal[sorted], eigVec[:, sorted]

    # Take transpose of the eigen Vectors and the data-feture matrix.
    # This is to find the projections of the images on Principal Components.
    eigvecTransposed = np.transpose(eigVec)
    arrTranspose = np.transpose(arr)

    # Find the projections by the 
    newData = (np.matmul(eigvecTransposed, arrTranspose))
    newDataTranspose = np.transpose(newData)
    with open("C://newData.csv", mode='w', newline='') as csvfile:
        _csvWriter = csv.writer(csvfile, delimiter=",")
        for i,row in enumerate(newDataTranspose):
            _csvWriter.writerow([[i]," | ",np.real(row)])

    print("\n--- %s minutes to execute --- \n" % round((time.clock() - start_time)/60,2))
    return newDataTranspose


def euclideanDistance(_inputFeatureVectorArray, _imgFeatureArray):
    sumOfSqrOfDiff = 0 
    for a,b in zip(_inputFeatureVectorArray,_imgFeatureArray):
        sumOfSqrOfDiff += (a-b)**2
    distance = math.sqrt(sumOfSqrOfDiff)
    return distance


def similarity_measure(_imageid, _collName, _k, arr, _m, _reductionTechnique):
    pcaFV = []
    if _reductionTechnique == "pca":
            pcaFV = pca(arr,_k)
    arrImgNames = []
    imgFVIndex = []
    _dict = {}
    _dist = 0
    for i,rec in enumerate(_collName.find()):
        if _imageid == (rec["ImageId"]):
            imgFVIndex = i
        arrImgNames.append(rec["ImageId"])
    for i,row in enumerate(pcaFV):
        _dist = euclideanDistance(pcaFV[imgFVIndex], pcaFV[i]) 
        _dict[arrImgNames[i]] =  1/(1+_dist)
    _result = sorted(_dict.items(), key=operator.itemgetter(1), reverse=True)
    for index,res in enumerate(_result):
        print(index, ". ", res[0], " | ", res[1])
        if index > _m:
            break

if __name__ == "__main__":
    # _task = int(input("Enter task(1/2): "))
    _task = 1
    # _modelName   = input("Enter model (cm/lbp/hog): ")
    _modelName   = "cm"
    # _k = int(input("Enter count of top latent semantics to compute: "))
    _k = 10
    # _reductionTechnique   = input("Enter reduction technique to use (pca) ")
    _reductionTechnique   = "pca"

    # _imageid = input("Enter image ID: ")
    _imageid = "Hand_0000025.jpg"
    
    # _m = int(input("Enter number of matching images:"))
    _m = 10

<<<<<<< HEAD
>>>>>>> commit latest in pca
=======
    mongo_client = MongoClient()
    _db = mongo_client.features
    # New array to store data-feature vectors
    arr = []
    # arr = [[1,2,3],[4,5,6],[17,18,19],[20,21,22]]
    # New array for image names from database
    # arrImgNames = []
    for i,rec in enumerate(_db[_modelName].find()):
        # Build array of data-feature vectors
        arr.append(rec["featureVector"])
        # Build array of image names
        # arrImgNames.append(rec["ImageId"])
>>>>>>> commit intermediate code for pca

    if _task == 1:
        if _reductionTechnique == "pca":
            pca(arr,_k)
    elif _task == 2:
        similarity_measure(_imageid, _db[_modelName], _k, arr, _m, _reductionTechnique)
