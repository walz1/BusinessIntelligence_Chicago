# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns

dataset = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])

## Do some Data cleaning
# Colums removed because we do not need them
dataset = dataset.drop(columns=['Unnamed: 0', 'ID', 'Case Number'])
# Colums removed because they contain to many missing values
dataset = dataset.drop(columns=['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Location'])
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()

## Parse Dates in File and specify format to speed up the operation
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
dataset['Updated On'] = pd.to_datetime(dataset['Updated On'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

## Do some value parsing
dataset['IUCR'] = dataset['IUCR'].astype('category')
dataset['Primary Type'] = dataset['Primary Type'].astype('category')
dataset['Location Description'] = dataset['Location Description'].astype('category')
dataset['Beat'] = dataset['Beat'].astype('int16')
dataset['District'] = dataset['District'].astype('int16')
dataset['Ward'] = dataset['Ward'].astype('int16')
dataset['Community Area'] = dataset['Community Area'].astype('int16')
dataset['Year'] = dataset['Year'].astype('int16')

sns.pairplot(dataset[['Community Area', 'Arrest', 'Year', 'FBI Code']], hue='Year', palette="husl")



#print(beats)
# optional step to read community Data and name areas
#communityAreas = pd.read_csv('~/Documents/GitHub/BI/CommAreas.csv', usecols=['AREA_NUMBE', 'COMMUNITY']) # https://data.cityofchicago.org/w/3fqw-rq4x/3q3f-6823?cur=JvAqhevqjta&from=root
#dataset = pd.merge(dataset,
#                   communityAreas,
#                   left_on='Community Area',
#                   right_on='AREA_NUMBE',
#                   how='left')

#dataset = dataset.drop(labels=['Unnamed: 0',
#                               'X Coordinate',
#                               'Y Coordinate',
#                               'Location',], axis=1)

#numberOfRecords = len(dataset.index)

#print(dataset)
#crimesCommited = dataset['Primary Type'].value_counts() / dataset.shape[0]
#print(crimesCommited)

