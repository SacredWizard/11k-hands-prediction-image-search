#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:55:44 2019

@author: karishmajoseph
"""

from numpy import array
from sklearn.decomposition import TruncatedSVD
# define array
A = array([
	[1,1,1,0,0],
	[3,3,3,0,0],
	[4,4,4,0,0],
    [5,5,5,0,0],
    [0,2,0,4,4],
    [0,0,0,5,5],
    [0,1,0,2,2]])
print(A)
# svd
svd = TruncatedSVD(n_components=2)
svd.fit(A)
result = svd.transform(A)
print(result)