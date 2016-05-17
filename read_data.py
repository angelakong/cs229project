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
Y = df['stars']
Y[Y < 4.0] = 0
Y[Y >= 4.0] = 1

alc = df["attributes.Alcohol"].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])
alc.fillna(mode(alc).mode[0], inplace = True)

takeout = df["attributes.Take-out"]
takeout.fillna(mode(takeout).mode[0], inplace = True)
takeout = takeout.astype(int)

waiter = df["attributes.Waiter Service"]
waiter.fillna(mode(waiter).mode[0], inplace = True)
waiter = waiter.astype(int)

casual_ambience = df["attributes.Ambience.casual"]
casual_ambience.fillna(mode(casual_ambience).mode[0], inplace = True)
casual_ambience = casual_ambience.astype(int)

frames = numpy.column_stack((takeout, waiter, casual_ambience, alc))
X = frames[1:1000]
Y = Y[1:1000]

fignum = 1
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel = kernel, gamma = 2)
    clf.fit(X,Y)
    plt.figure(fignum, figsize = (4,3))
    plt.clf()
plt.show()
    
    









