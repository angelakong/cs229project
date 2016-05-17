# -*- coding: utf-8 -*-
"""
Created on Tue May 17 02:19:31 2016

@author: angelakong
"""

import json
import numpy
import pandas
from sklearn import svm
from numpy import array
from array import *
import csv
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import mode
import matplotlib.pyplot as plt


df = pandas.read_csv('yelp_academic_dataset_business.csv', skip_blank_lines=True)

# Classification vector
y = df['stars']
y[y < 4.0] = 0
y[y >= 4.0] = 1

alc = df["attributes.Alcohol"].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])
alc.fillna(mode(alc).mode[0], inplace = True)

takeout = df["attributes.Take-out"]
takeout.fillna(0, inplace = True)
takeout = takeout.astype(int)

waiter = df["attributes.Waiter Service"]
# If restaurant does not list a benefit, we assume that it lacks it
waiter.fillna(0, inplace = True)

# Fill the empty cells with the mode of category (one that appears the most)
#waiter.fillna(mode(waiter).mode[0], inplace = True)
waiter = waiter.astype(int)

casual_ambience = df["attributes.Ambience.casual"]
casual_ambience.fillna(0, inplace = True)
#casual_ambience.fillna(mode(casual_ambience).mode[0], inplace = True)
casual_ambience = casual_ambience.astype(int)

frames = numpy.column_stack((takeout, waiter, casual_ambience, alc))
X = frames[1:500]
print(X)
ytrain = y[1:500]

print(frames[2000])

#for kernel in ('linear', 'poly', 'rbf'):

clf = svm.SVC()
clf.fit(X,ytrain)

print(frames[1000])

numTestData = 1000
ytest = y[1000:2000]
print(ytest.shape)

predictions = []
predictions = clf.predict(frames[1000:2000])
print(predictions.shape)

    
error = sum(ytest * predictions <= 0) / numTestData
print("Test error for SVM: " + str(error))

#clf = svm.SVC(kernel = 'linear', gamma = 2)
#clf.fit(X,y)
#plt.figure(fignum, figsize = (4,3))
#    
#w = clf.coef_[0]
#    
#a = -w[0] / w[1]
#    
#xx = numpy.linspace(0,12)
#yy = a * xx - clf.intercept_[0] / w[1]
#h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
#plt.scatter(X[:, 0], X[:, 1], c = y)
#plt.legend()
#plt.show()



    
    









