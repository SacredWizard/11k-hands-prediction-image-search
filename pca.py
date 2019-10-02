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
from sklearn.decomposition import PCA
=======


>>>>>>> commit intermediate code for pca

mongo_client = MongoClient()
db = mongo_client.features
coll = db.hog
<<<<<<< HEAD
arr = [[1,2,3],[3,2,1],[2,2,2]]

# for i,rec in enumerate(coll.find()):
#     arr.append(rec["featureVector"])

# print(arr)
print(np.shape(arr))

#####################
meanmatrix = np.mean(arr, axis = 0)
# print(meanmatrix)
=======
arr = []
for i,rec in enumerate(coll.find()):
    arr.append(rec["featureVector"])
print(arr)
print(np.shape(arr))

meanmatrix = np.mean(arr, axis = 0)
print(meanmatrix)
>>>>>>> commit intermediate code for pca
meanNormalized = arr
print(np.shape(meanmatrix))
for i,row in enumerate(arr):
    meanNormalized[i] = np.array(arr[i] - meanmatrix)
# arr = np.var(arr, axis = 1)
<<<<<<< HEAD
# print(meanNormalized)
print(np.shape(meanNormalized))
covMatrix = np.cov(arr, rowvar=False)

print ("xyz")
print(np.shape(covMatrix))
# print(covMatrix)
eigVal,eigVec = np.linalg.eig(covMatrix)

print("Eigen values", np.shape(eigVal))
# print(eigVal)
print("Eigen vectors", np.shape(eigVec))
# print(eigVec)
=======
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
>>>>>>> commit intermediate code for pca

sorted = eigVal.argsort()[::-1]
eigVal,eigVec = eigVal[sorted], eigVec[:, sorted]

print("after sort")
print("Eigen values", np.shape(eigVal))
<<<<<<< HEAD
# print(eigVal)
print("Eigen vectors", np.shape(eigVec))
# print(eigVec)
prinComponents = np.dot(meanNormalized,eigVec)
# print(prinComponents[:][:5])
print("Principal components", np.shape(prinComponents))
#################

# print(arr)
# pca = PCA(n_components = 3,copy=False)
# decompsedArr = pca.fit(arr)
# print("decompsedArr",decompsedArr)
# transformArr = decompsedArr.transform(arr)
# print("transformArr",transformArr)
=======
print(eigVal)
print("Eigen vectors", np.shape(eigVec))
print(eigVec)
prinComponents = np.dot(meanNormalized,eigVec)
print(prinComponents[:][:5])
print("Principal components", np.shape(prinComponents))
>>>>>>> commit intermediate code for pca
