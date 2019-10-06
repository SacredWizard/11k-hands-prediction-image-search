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
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from skimage import io, color
from skimage.feature import hog, local_binary_pattern
from skimage.transform import downscale_local_mean, rescale
<<<<<<< HEAD



mongo_client = MongoClient()
db = mongo_client.features
coll = db.hog
arr = []
for i,rec in enumerate(coll.find()):
    arr.append(rec["featureVector"])
print(arr)
print(np.shape(arr))

meanmatrix = np.mean(arr, axis = 0)
print(meanmatrix)
meanNormalized = arr
print(np.shape(meanmatrix))
for i,row in enumerate(arr):
    meanNormalized[i] = np.array(arr[i] - meanmatrix)
# arr = np.var(arr, axis = 1)
print(meanNormalized)
print(np.shape(meanNormalized))
covMatrix = np.cov(arr, rowvar=False)
print(np.shape(covMatrix))
print(covMatrix)
eigVal,eigVec = np.linalg.eig(covMatrix)

print("Eigen values", np.shape(eigVal))
print(eigVal)
print("Eigen vectors", np.shape(eigVec))
print(eigVec)
=======
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import csv

def pca(_collName,_k):
    # New array to store data-feature vectors
    arr = []
    # arr = [[110,12,13],[14,15,16],[17,18,19]]
    # New array for image names from database
    arrImgNames = []
    for i,rec in enumerate(_collName.find()):
        # Build array of data-feature vectors
        arr.append(rec["featureVector"])
        # Build array of image names
        arrImgNames.append(rec["ImageId"])

    # print("arr\n",arr)
    # print(np.shape(arr))

    # Normalize data-feature vector array along column
    meanNormalized = normalize(arr,axis=0,copy=False)
    # print("meanNormalized\n" ,meanNormalized)
    # Again normalize feature vector array along row
    meanNormalized = normalize(meanNormalized,axis=1,copy=False)
    # print("meanNormalized\n" ,meanNormalized)
    # print(np.shape(meanNormalized))
    # Calculate the covariance matrix, rowvar=False means calculate covariance of feature vectors 
    covMatrix = np.cov(meanNormalized, rowvar=False)

    # print ("xyz")
    # print(np.shape(covMatrix))
    # print("covMatrix\n",covMatrix)
    # Calculate eigen values and eigen vectors of the covariane matrix of feature vectors
    eigVal,eigVec = np.linalg.eigh(covMatrix)

    # print("Eigen values", np.shape(eigVal))
    # print(eigVal)
    # print("Eigen vectors", np.shape(eigVec))
    # print(eigVec)

    # Get the index of original eigen values in ascending order
    sorted = eigVal.argsort()[::-1]

    # print("after sort")
    with open("C://topLatentSemantics.csv", mode='w', newline='') as csvfile:
        _csvWriter = csv.writer(csvfile, delimiter=",")
        for i in sorted:
            _csvWriter.writerow([i,eigVal[i]])
    print("Index of original feature vector | weight")
    for i in range(len(sorted)-1,len(sorted)-_k-1,-1):
        print(sorted[i],eigVal[sorted[i]])
>>>>>>> cb97e9eac1c8db0556b453bd52ad11966eccda66

    # print("sorted\n", sorted)
    # sort the eigen values and eigen vectors based on increasing eigen values
    eigVal,eigVec = eigVal[sorted], eigVec[:, sorted]

    # print("Eigen values", np.shape(eigVal))
    # print(eigVal)
    # print("Eigen vectors", np.shape(eigVec))
    # print(eigVec)
        # prinComponents = np.dot(meanNormalized,eigVec)
        # print("Principal components", np.shape(prinComponents))
        # print(prinComponents[:][:5])
        #################

<<<<<<< HEAD
# print(arr)
# pca = PCA(n_components = 3,copy=False)
# decompsedArr = pca.fit(arr)
# print("decompsedArr",decompsedArr)
# transformArr = decompsedArr.transform(arr)
# print("transformArr",transformArr)
print(eigVal)
print("Eigen vectors", np.shape(eigVec))
print(eigVec)
prinComponents = np.dot(meanNormalized,eigVec)
print(prinComponents[:][:5])
print("Principal components", np.shape(prinComponents))
=======
        # print(arr)
        # pca = PCA(n_components = 3,copy=False)
        # decompsedArr = pca.fit(arr)
        # print("decompsedArr",decompsedArr)
        # transformArr = decompsedArr.transform(arr )
        # print("transformArr",transformArr)
        
if __name__ == "__main__":
    _modelName   = input("Enter model (cm/lbp/hog): ")
    _k   = int(input("Enter count of top latent semantics to compute: "))
    _reductionTechnique   = input("Enter reduction technique to use (pca) ")
    mongo_client = MongoClient()
    _db = mongo_client.features
    if(_reductionTechnique == "pca"):
        pca(_db[_modelName],_k)
>>>>>>> cb97e9eac1c8db0556b453bd52ad11966eccda66
