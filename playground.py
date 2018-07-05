# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
"""
dataset_04 = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])

dataset_07 = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2005_to_2007.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])

dataset_11 = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2008_to_2011.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])

dataset_17 = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2012_to_2017.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])
"""
dataset = pd.concat([dataset_04, dataset_07, dataset_11, dataset_17])


## Do some Data cleaning
# Colums removed because we do not need them
dataset = dataset.drop(columns=['Unnamed: 0', 'Community Area', 'District', 'FBI Code', 'Updated On', 'Year', 'Location Description', 'IUCR'])
# Colums removed because they contain to many missing values
dataset = dataset.drop(columns=['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location'])
dataset = dataset.drop_duplicates()
dataset = dataset.drop(columns=['ID', 'Case Number'])
dataset = dataset.dropna()

## Parse Dates in File and specify format to speed up the operation
# we create additional columns for single date attributes
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
dataset['Date-year'] = dataset['Date'].dt.year
dataset['Date-month'] = dataset['Date'].dt.month
dataset['Date-day'] = dataset['Date'].dt.day
dataset['Date-hour'] = dataset['Date'].dt.hour
dataset['Date-minute'] = dataset['Date'].dt.minute
dataset['Day-of-week'] = dataset['Date'].dt.dayofweek


## Convert Primary Type Category to binary encoded information --> Folienset 7 31
binStyle = LabelBinarizer()
binPrimaryType = binStyle.fit_transform(dataset['Primary Type'])
binPrimaryType = pd.DataFrame(binPrimaryType, columns=binStyle.classes_)

dataset = pd.concat([dataset, binPrimaryType], axis=1, join_axes=[dataset.index])
# we can think about dropping the primary type now, as we have the information strored elsewhere


## Normalize Date-minute to 0 or 30
dataset['Date-minute'] = dataset['Date-minute'].apply(lambda x: x >= 30 and 30 or 0)




"""
### -- Beginn of drawing
fig, ((axis1,axis2,axis3),(axis4,axis5,axis6)) = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(18,6)

sns.countplot(data=dataset, x='Date-year', ax=axis1)
sns.countplot(data=dataset, x='Date-month', ax=axis2)
sns.countplot(data=dataset, x='Date-day', ax=axis3)
sns.countplot(data=dataset, x='Date-hour', ax=axis4)
sns.countplot(data=dataset, x='Date-minute', ax=axis5)

fig, (axis1,axis2) = plt.subplots(nrows=2, ncols=1, figsize=(18,4))
sns.countplot(data=dataset, x='Date-hour', ax=axis1)
sns.countplot(data=dataset, x='Date-minute', ax=axis2)
plt.show()
### -- End of drawing

crimeTypeWard = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Primary Type', 'Ward'] ).size()}).reset_index()
"""

dataset['Beat'] = pd.get_dummies(dataset.Beat)
tmp = pd.get_dummies(dataset.Ward)
dataset['Ward'] = pd.get_dummies(dataset.Ward)

#print(dataset)
#crimesCommited = dataset['Primary Type'].value_counts() / dataset.shape[0]
#print(crimesCommited)

from sklearn import tree
import pandas as pd

dataset_b = dataset.sample(5000000) #random build sample n=100
dataset_t = dataset.loc[~dataset.index.isin(dataset_b.index)] #test sample n=50

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(dataset_b[['Beat', 'Ward', 'Arrest', 'Domestic', 'Date-month', 'Date-day', 'Date-hour', 'Date-minute']], dataset_b['Primary Type'])

print("Model Accuracy:")
print(clf.score(dataset_t[['Beat', 'Ward', 'Arrest', 'Domestic', 'Date-month', 'Date-day', 'Date-hour', 'Date-minute']], dataset_t['Primary Type']))

dataset_b = dataset.sample(5000000) #random build sample n=100
dataset_t = dataset.loc[~dataset.index.isin(dataset_b.index)] #test sample n=50

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(dataset_b[['Beat', 'Ward', 'Arrest', 'Domestic', 'Day-of-week']], dataset_b['Primary Type'])

print("Model Accuracy:")
print(clf.score(dataset_t[['Beat', 'Ward', 'Arrest', 'Domestic', 'Day-of-week']], dataset_t['Primary Type']))

dataset_b = dataset.sample(5000000) #random build sample n=100
dataset_t = dataset.loc[~dataset.index.isin(dataset_b.index)] #test sample n=50

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(dataset_b[['Beat', 'Ward', 'Arrest', 'Domestic']], dataset_b['Primary Type'])

print("Model Accuracy:")
print(clf.score(dataset_t[['Beat', 'Ward', 'Arrest', 'Domestic']], dataset_t['Primary Type']))
