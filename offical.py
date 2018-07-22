# -*- coding: utf-8 -*-
"""
Here we need to write something general about our code

"""
##############################################################################
######################## Import section ######################################
##############################################################################

import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from sklearn import tree, svm, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import graphviz 

from sklearn.naive_bayes import BernoulliNB

## Global Settings for multiprocessing
NUM_PARTITIONS = 10 #number of partitions to split dataframe
NUM_CORES = 4 #number of cores on your machine
CORRELATION_SAMPLE_SIZE = 0.25
##############################################################################
######################## Method section ######################################
##############################################################################

def fix_block(toFix):
    """
    Here we need to describe the method
    """
    toFix = toFix.split()
    if (len(toFix) >= 3 and toFix[2][0].isdigit() and toFix[2][len(toFix[2]) - 1].isdigit()):
        if toFix[2][len(toFix[2]) - 1] == '1':
            toFix[2] = toFix[2] + 'ST'
        elif toFix[2][len(toFix[2]) - 1] == '2':
            toFix[2] = toFix[2] + 'ND'
        elif toFix[2][len(toFix[2]) - 1] == '3':
            toFix[2] = toFix[2] + 'RD'
        else:
            toFix[2] = toFix[2] + 'TH'
           
    if len(toFix) > 3:
        if toFix[3] == 'STREET':
            toFix[3] = 'ST'
        elif toFix[3] == 'PLACE':
            toFix[3] = 'PL'
        elif toFix[3] == 'BL' or toFix[3] == 'BLV':
            toFix[3] = 'BLVD'
        elif toFix[3] == 'PW':
            toFix[3] = 'PWKY'
        elif toFix[3] == 'AV':
            toFix[3] = 'AVE'

    return ' '.join(toFix)

def basic_kendall_corr(data):
    data = data.sample(frac=CORRELATION_SAMPLE_SIZE)
    return data.corr('kendall')

def corr_multidimension_ptype(data, dataPrimary):
    res = [0 for x in range(0, len(data.columns))]
    i = 0
    for column in data:
        tmpDataFrame = pd.concat([data[column], dataPrimary], axis=1, join_axes=[data.index])
        tmpDataFrame = tmpDataFrame.sample(frac=CORRELATION_SAMPLE_SIZE)
        res[i] = tmpDataFrame.corr('kendall')
        i+=1
    return res


## http://www.racketracer.com/2016/07/06/pandas-in-parallel/
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, NUM_PARTITIONS)
    pool = Pool(NUM_CORES)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

##############################################################################
######################## Data loading ########################################
##############################################################################

print("Started section: Data loading")
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
#"""
#dataset = pd.concat([dataset_04, dataset_07, dataset_11, dataset_17])

print("Finished section: Data loading")

##############################################################################
######################## Data preparation ####################################
##############################################################################

print("Started section: Data preparation")
######################## Data selection ######################################
"""
# Colums removed because we do not need them
dataset = dataset.drop(columns=['Unnamed: 0', 'Case Number', 'ID', 'Community Area', 'Description', 'FBI Code', 'Updated On', 'Year', 'Location Description', 'IUCR'])

# Colums removed because they contain to many missing values
dataset = dataset.drop(columns=['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location'])
#"""
print("Finished subsection: Data selection")

######################## Data cleaning #######################################
"""

#dataset = dataset.drop_duplicates(subset=['Case Number', 'ID'])
# Fix Blocks to contain ST, ND, RD, TH
def parallel_block_fix(data):
    data = data.apply(lambda x: fix_block(x))
    return data

dataset['Block'] = parallelize_dataframe(dataset['Block'], parallel_block_fix)


tmpBlockWardDict = dataset[['Block', 'Ward']].drop_duplicates().dropna().set_index('Block')
tmpBlockWardDict = tmpBlockWardDict.to_dict()

def parallel_ward_fix(data):
    return data.apply(lambda row: tmpBlockWardDict['Ward'].get(row['Block']), axis=1)

dataset['Ward'] = parallelize_dataframe(dataset, parallel_ward_fix)
dataset = dataset.dropna()
#"""
print("Finished subsection: Data cleaning")

######################## Data construction & formatting ######################
"""
## Parse Dates in File and specify format to speed up the operation
# we create additional columns for single date attributes

def parallel_dates(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data['Date-year'] = data['Date'].dt.year
    data['Date-month'] = data['Date'].dt.month
    data['Date-day'] = data['Date'].dt.dayofweek
    data['Date-hour'] = data['Date'].dt.hour
    data['Date-minute'] = data['Date'].dt.minute

    # Normalize Date-minute to 0 or 30
    data['Date-minute'] = data['Date-minute'].apply(lambda x: x >= 30 and 30 or 0)
    return data

dataset = parallelize_dataframe(dataset, parallel_dates)
dataset = dataset.drop(columns=['Date'])

# Convert categorials to binary encoded information --> Folienset 7 31
binWard = pd.get_dummies(dataset.Ward)
dataset = pd.concat([dataset, binWard], axis=1, join_axes=[dataset.index])

binDistrict = pd.get_dummies(dataset.District)
dataset = pd.concat([dataset, binDistrict], axis=1, join_axes=[dataset.index])

binBeat= pd.get_dummies(dataset.Beat)

binYear = pd.get_dummies(dataset['Date-year'])
binMonth = pd.get_dummies(dataset['Date-month'])
binDay = pd.get_dummies(dataset['Date-day'])
binHour = pd.get_dummies(dataset['Date-hour'])
binMinute = pd.get_dummies(dataset['Date-minute'])

binPrimaryType = pd.get_dummies(dataset['Primary Type'])

#"""
print("Finished subsection: Data construction & formatting")
print("Finished section: Data preparation")

##############################################################################
######################## Data understanding ##################################
##############################################################################
print("Started section: Data understanding")
######################## Explore Data ######'#################################

"""
### Basic python drawing. For more elaborated visualization review our attached
### HTML file (https://walz1.github.io/BusinessIntelligence_Chicago/)
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

crimeBeat = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Beat' ] ).size()}).reset_index()
crimeWard = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Ward' ] ).size()}).reset_index()
crimeType = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Primary Type' ] ).size()}).reset_index()
crimeTypeWard = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Primary Type', 'Ward' ] ).size()}).reset_index()
crimeTypeBeat = pd.core.frame.DataFrame({'count' : dataset.groupby( [ 'Primary Type', 'Beat' ] ).size()}).reset_index()


### Correlation Analyses

corrWardPtype = corr_multidimension_ptype(binWard, binPrimaryType)
corrDistrictPtype = corr_multidimension_ptype(binDistrict, binPrimaryType)
corrArrestPtype = basic_kendall_corr(pd.concat([dataset['Arrest'], binPrimaryType], axis=1, join_axes=[dataset.index]))
corrDomesticPtype = basic_kendall_corr(pd.concat([dataset['Domestic'], binPrimaryType], axis=1, join_axes=[dataset.index]))

corrYearPtype = corr_multidimension_ptype(binYear, binPrimaryType)
corrMonthPtype = corr_multidimension_ptype(binMonth, binPrimaryType)
corrDayPtype = corr_multidimension_ptype(binDay, binPrimaryType)
corrHourPytpe = corr_multidimension_ptype(binHour, binPrimaryType)
corrMinutePtype = corr_multidimension_ptype(binMinute, binPrimaryType)
"""
print("Finished section: Data understanding")

##############################################################################
######################## Modeling ############################################
##############################################################################
print("Started section: Modeling")

######################## Modeling Technique ##################################
### Decision Tree

## This will predict all categories within one run 
"""
print("Decision Tree")
x = dataset.drop(columns=['Primary Type', 'Block', 'Arrest', 'Domestic', 'Ward'])
y = binPrimaryType
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(x_train, y_train)
predictDTree = clf.predict(x_test)
scoreDTree = clf.score(x_test, y_test)
print("Model Accuracy: ", scoreDTree)
#"""

### Logistic Regression
#"""
print("Logistic Regression")
x = pd.concat([binWard, binDistrict, binBeat, binMonth, binDay, binHour, binMinute], axis=1, join_axes=[binWard.index])
y = binPrimaryType['THEFT']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

logRegression = LogisticRegression()
logRegression.fit(x_train, y_train)
predictLogRegression = logRegression.predict(x_test)
scoreLogRegression = logRegression.score(x_test, y_test)
print("Model Accuracy: ", scoreLogRegression)
#"""

### Bernoullo Naive Bayes
"""
print("Bernoulli Naive Bayes")
x = pd.concat([binWard, binBeat, binMonth, binDay, binHour, binMinute], axis=1, join_axes=[binWard.index])
y = binPrimaryType["THEFT"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = BernoulliNB()
clf.fit(x_train, y_train)
predictNB = clf.predict(x_test)
scoreNB = clf.score(x_test, y_test)
print("Model Accuracy: ", scoreNB)

"""