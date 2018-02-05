#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:54:06 2018

@author: Smriti
"""

from numpy import *
#import kNN
from matplotlib import *
from matplotlib.pyplot import * 
from os import *
import operator

def classify_numerical(inX, dataSet, labels, k):
    # number of rows in the dataSet. Each attribute is a column of the dataSet matrix 
    dataSetSize = dataSet.shape[0]
    # this creates a matrix of distances from inX to each training sample in dataSet.
    # it is done by copying inX into each row of a matrix the number of rows as the dataSet 
    # the dataSet matrix is then subtracted from the new matrix of repeated inX
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # the difference matrix is then squared, and the Eucidean distances matrix is calculated 
    # by summing the squares of the differences over all attributes and taking the square root 
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # the distances are now sorted, and the indices of the sorted values
    # are stored in sortedDistIndicies
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        # keep track of the class label for each of k nearest neighbors
        voteIlabel = labels[sortedDistIndicies[i]]
        # increment the item in the dictionary referenced by the key
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # find the item in the dictionary with the most votes
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]

def classify_nominal(inX, dataSet, labels, k):
    # number of rows in the dataSet. Each attribute is a column of the dataSet matrix 
    dataSetSize = dataSet.shape[0]
    # this creates a matrix of distances from inX to each training sample in dataSet.
    # it is done by copying inX into each row of a matrix the number of rows as the dataSet 
    test_mat = tile(inX, (dataSetSize,1))
    test_mat = (dataSet != test_mat)    
    distances = average(test_mat,axis=1)
    
    # the distances are now sorted, and the indices of the sorted values
    # are stored in sortedDistIndicies
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        # keep track of the class label for each of k nearest neighbors
        voteIlabel = labels[sortedDistIndicies[i]]
        # increment the item in the dictionary referenced by the key
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # find the item in the dictionary with the most votes
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]

# training data
rawTrainingData = loadtxt('poker-hand-training.txt', delimiter = ',')
rawTestingData = loadtxt('poker-hand-testing.txt', delimiter = ',')

# Separate the data and the labels
data = rawTrainingData[:,0:10]
labels = rawTrainingData[:,10]

# k to try
k_try = [5, 25, 45, 65, 85, 105, 205, 305, 405, 505]

# Case 2: Considering all the data as nominal
result_nom = zeros((10000,len(k_try)))
# classify each point in the testing data (take 10000 testing points)
for i1 in range(len(k_try)):
    for i in range(10000):
        result_nom[i,i1] = classify_nominal(rawTestingData[i,0:10], data, 
                 labels, k_try[i1]) 

error_nom = zeros((len(k_try),1))
for i in range(len(k_try)):
    error_nom[i] = mean( result_nom[:,i] != rawTestingData[:10000,10] )

error_nom = error_nom*100  
print "Error assuming nominal"
print error_nom

# Case 1: Scale the data
min_array = amin(data, axis = 0)
max_array = amax(data, axis = 0)
for i in range(len(data)):
    for j in range(10):
        data[i,j] = (data[i,j] - min_array[j])/(max_array[j]-min_array[j])

for i in range(10000):
    for j in range(10):
        rawTestingData[i,j] = (rawTestingData[i,j] - min_array[j])/(max_array[j]-min_array[j])
        
# Case 1: Considering all the data as numerical
result_num = zeros((10000,len(k_try)))
for i1 in range(len(k_try)):
    for i in range(10000):
        result_num[i,i1] = classify_numerical(rawTestingData[i,0:10], data, 
                  labels, k_try[i1])

error_num = zeros((len(k_try),1))
for i in range(len(k_try)):
    error_num[i] = mean( result_num[:,i] != rawTestingData[:10000,10] )

error_num = error_num*100  
print "Error assuming numerical"
print error_num                    