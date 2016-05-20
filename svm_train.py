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
# There are 77445 training examples, 98 possible features

features = df.columns.values

#for index, feature in enumerate(features):
#    if "categories" in feature:
#        print index
        
category_index = 20

# Classification vector
y = df['stars']
y[y < 4.0] = 0
y[y >= 4.0] = 1

# Remove the non-restaurants from the data
for i, category in enumerate(df["categories"]):
    if "u'Restaurants" not in category:
        df.drop(df.index[i])

print(df.shape)

# Preprocess the data for modeling
# Prepare for messy code
for f in features:
    # Special code for categories
    # Drop hours for now (easier to process)
    delete = ["hours", "business_id", "stars", "type", "categories", "name", "full_address", "BYOB", "neighborhoods"]
    if any(word in f for word in delete):
        df.drop(f, axis=1, inplace=True)
    else:
        if "Attire" in f:
            df[f] = df[f].replace(['casual', 'dressy'], [0,1])
        
        if "Noise" in f:
            df[f] = df[f].replace(['quiet', 'average', 'loud', 'very_loud'], [0,1,2,3])
        
        if "Wi-Fi" in f:
            df[f] = df[f].replace(['no', 'free'], [0,1]) 

        if "Alcohol" in f:
            df[f] = df[f].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])

        df[f].fillna(0, inplace = True)
        if "FALSE" in df[f].tolist(): #Convert true/false to integer values
            df[f] = df[f].astype(int)
        

#alc = df["attributes.Alcohol"].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])
#alc.fillna(mode(alc).mode[0], inplace = True)
#
#takeout = df["attributes.Take-out"]
#takeout.fillna(0, inplace = True)
#takeout = takeout.astype(int)
#
#waiter = df["attributes.Waiter Service"]
## If restaurant does not list a benefit, we assume that it lacks it
#waiter.fillna(0, inplace = True)
#
## Fill the empty cells with the mode of category (one that appears the most)
##waiter.fillna(mode(waiter).mode[0], inplace = True)
#waiter = waiter.astype(int)
#
#casual_ambience = df["attributes.Ambience.casual"]
#casual_ambience.fillna(0, inplace = True)
##casual_ambience.fillna(mode(casual_ambience).mode[0], inplace = True)
#casual_ambience = casual_ambience.astype(int)

frames = numpy.column_stack((takeout, waiter, casual_ambience, alc))
X = frames[1:500]
print(X)
ytrain = y[1:500]

#for kernel in ('linear', 'poly', 'rbf'):

clf = svm.SVC()
clf.fit(X,ytrain)


numTestData = 1000
ytest = y[1000:2000]

predictions = []
predictions = clf.predict(frames[1000:2000])

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