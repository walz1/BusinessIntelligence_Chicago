# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

dataset = pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])#, nrows=1602850)

# Parse Dates in File and specify format to speed up the operation
dataset["Date"] = pd.to_datetime(dataset["Date"], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
dataset['Updated On'] = pd.to_datetime(dataset['Updated On'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Parse some columns to type 'category'. This saves ~200MB of space
dataset['Primary Type'] = dataset['Primary Type'].astype('category')
dataset['Location Description'] = dataset['Location Description'].astype('category')


#beats =  pd.read_csv('~/Documents/GitHub/BusinessIntelligence_Chicago/Beat.csv',
#                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
#                      na_values=[''])

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

