# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import matplotlib.pyplot as plat

dataset = pd.read_csv('~/Documents/GitHub/BI/dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''], nrows=1000, parse_dates=['Date', 'Updated On'])

# optional step to read community Data and name areas
#communityAreas = pd.read_csv('~/Documents/GitHub/BI/CommAreas.csv', usecols=['AREA_NUMBE', 'COMMUNITY']) # https://data.cityofchicago.org/w/3fqw-rq4x/3q3f-6823?cur=JvAqhevqjta&from=root
#dataset = pd.merge(dataset,
#                   communityAreas,
#                   left_on='Community Area',
#                   right_on='AREA_NUMBE',
#                   how='left')

dataset = dataset.drop(labels=['Unnamed: 0',
                               'X Coordinate',
                               'Y Coordinate',
                               'Location',], axis=1)
numberOfRecords = len(dataset.index)

print
#crimesCommited = dataset['Primary Type'].value_counts() / dataset.shape[0]
#print(crimesCommited)