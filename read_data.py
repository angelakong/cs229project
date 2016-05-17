import json
import numpy
from numpy import array
from array import *
import csv
from sklearn import svm


input = open('yelp_academic_dataset_business.csv', 'r');
features = csv.reader(input)

X = numpy.array(list(features))
print(X)

f = open("yelp_academic_dataset_business.json", "r")
lines = f.readlines()
reviews = []
Y = [0]*len(lines)

for i in range(0, len(lines)):
    business_example = json.loads(lines[i])
    if 'Restaurants' in business_example['categories'] and 'Las Vegas' in business_example['city']:
        
        if business_example['stars'] >= 4.0:
            Y[i] = 1
        reviews.append(business_example)
        


print(Y)

        
