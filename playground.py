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

## Global Settings for multiprocessing
num_partitions = 10 #number of partitions to split dataframe
num_cores = 4 #number of cores on your machine

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
    #data = data.sample(frac=0.5)
    return data.corr('kendall')

## http://www.racketracer.com/2016/07/06/pandas-in-parallel/
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
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
dataset = pd.concat([dataset_04, dataset_07, dataset_11, dataset_17])

print("Finished section: Data loading")

##############################################################################
######################## Data preparation ####################################
##############################################################################

print("Started section: Data preparation")
######################## Data selection ######################################
#"""
# Colums removed because we do not need them
dataset = dataset.drop(columns=['Unnamed: 0', 'Case Number', 'ID', 'Community Area', 'Description', 'FBI Code', 'Updated On', 'Year', 'Location Description', 'IUCR'])

# Colums removed because they contain to many missing values
dataset = dataset.drop(columns=['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location'])
#"""
print("Finished subsection: Data selection")

######################## Data cleaning #######################################
#"""
# Fix Blocks to contain ST, ND, RD, TH
def parallel_block_fix(data):
    data = data.apply(lambda x: fix_block(x))
    return data

abc = dataset['Block'].dropna().drop_duplicates()
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
#"""
## Parse Dates in File and specify format to speed up the operation
# we create additional columns for single date attributes

def parallel_dates(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
 #   data['Date-year'] = data['Date'].dt.year
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

binBeat= pd.get_dummies(dataset.Beat)
#dataset = pd.concat([dataset, binBeat], axis=1, join_axes=[dataset.index])

#binMonth = pd.get_dummies(dataset['Date-month'])
#dataset = pd.concat([dataset, binMonth], axis=1, join_axes=[dataset.index])

#binDay = pd.get_dummies(dataset['Date-day'])
#dataset = pd.concat([dataset, binDay], axis=1, join_axes=[dataset.index])

#binHour = pd.get_dummies(dataset['Date-hour'])
#dataset = pd.concat([dataset, binHour], axis=1, join_axes=[dataset.index])

binMinute = pd.get_dummies(dataset['Date-minute'])
dataset = pd.concat([dataset, binMinute], axis=1, join_axes=[dataset.index])

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
### HTML file.
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

"""
### Correlation Analyses

def corr_ward_ptype():
    res = [0 for x in range(0, len(binWard))]
    for i in range (1, len(binWard) + 1):
        tmpDataFrame = pd.concat([binWard[float(i)], binPrimaryType], axis=1, join_axes=[binWard.index])
        tmpDataFrame = tmpDataFrame.sample(frac=0.25)
        res[i-1] = tmpDataFrame.corr('kendall')
    return res

def corr_date0_ptype(data):
    res = [0 for x in range(0, len(data))]
    for i in range (0, len(res)):
        tmpDataFrame = pd.concat([data[i], binPrimaryType], axis=1, join_axes=[data.index])
        tmpDataFrame = tmpDataFrame.sample(frac=0.25)
        res[i] = tmpDataFrame.corr('kendall')
    return res

def corr_multidimension_ptype(data, dataPrimary):
    res = [0 for x in range(0, len(data.columns))]
    #for i in range (1, len(res) + 1):
    i = 0
    for column in data:
        tmpDataFrame = pd.concat([data[column], dataPrimary], axis=1, join_axes=[data.index])
        tmpDataFrame = tmpDataFrame.sample(frac=0.1)
        res[i] = tmpDataFrame.corr('kendall')
        i+=1
    return res

corrWardPtype = corr_multidimension_ptype(binWard)
#corrArrestPtype = basic_kendall_corr(pd.concat([dataset['Arrest'], binPrimaryType], axis=1, join_axes=[dataset.index]))
#corrDomesticPtype = basic_kendall_corr(pd.concat([dataset['Domestic'], binPrimaryType], axis=1, join_axes=[dataset.index]))

#corrMonthPtype = corr_date1_ptype(binMinute)
#corrDayPtype = corr_date1_ptype(binDay)
#corrHourPytpe = corr_date0_ptype(binHour)

#corrHourPtype = corr_date0_ptype(dataset['Date-hour'])

print("Finished section: Data understanding")

##############################################################################
######################## Modeling ############################################
##############################################################################
print("Started section: Modeling")

######################## Modeling Technique ##################################
### Decision Tree

## This will predict all categories within one run 
"""
x = dataset.drop(columns=['Ward', 'Primary Type', 'Block'])
y = binPrimaryType

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = tree.DecisionTreeRegressor()
cl_fit = clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print("Model Accuracy:")
print(clf.score(x_test, y_test))

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print("Model Accuracy:")
print(clf.score(x_test, y_test))
"""
#x_train = x_train.drop(columns=['Date-minute'])
#x_test = x_test.drop(columns=['Date-minute'])
#clf = tree.DecisionTreeClassifier()
#cl_fit = clf.fit(x_train, y_train)

#print("Model Accuracy:")
#print(clf.score(x_test, y_test))

#"""

### Logistic Regression
"""
x = dataset.drop(columns=['Arrest', 'Domestic', 'Ward', 'Primary Type', 'Block', 'District'])
y = binPrimaryType["THEFT"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

logRegression = LogisticRegression()
logRegression.fit(x_train, y_train)
predictions = logRegression.predict(x_test)
score = logRegression.score(x_test, y_test)    
cnf_matrix = confusion_matrix(y_test, predictions).ravel()
#"""

### SVM
"""
x = dataset.drop(columns=['Arrest', 'Domestic', 'Ward', 'Primary Type', 'Block', 'District'])
y = binPrimaryType["MOTOR VEHICLE THEFT"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

h=.02 # step size in the mesh
X = x_train
Y = y_train
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
svc     = svm.SVC(kernel='linear').fit(X, Y)
rbf_svc = svm.SVC(kernel='poly').fit(X, Y)
nu_svc  = svm.NuSVC(kernel='linear').fit(X,Y)
lin_svc = svm.LinearSVC().fit(X, Y)

# create a mesh to plot in
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with polynomial (degree 3) kernel',
          'NuSVC with linear kernel',
          'LinearSVC (linear kernel)']


pl.set_cmap(pl.get_cmap('jet'))

for i, clf in enumerate((svc, rbf_svc, nu_svc, lin_svc)):
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 2, i+1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.set_cmap(pl.get_cmap('Accent'))
    pl.contourf(xx, yy, Z)
    pl.axis('tight')

    # Plot also the training points
    pl.scatter(X[:,0], X[:,1], c=Y)

    pl.title(titles[i])

pl.axis('tight')
pl.show()
#"""
"""
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x.feature_names,  
                         class_names=dataset.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
"""