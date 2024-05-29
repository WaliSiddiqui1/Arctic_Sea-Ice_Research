#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:16:15 2024

@author: saminakashif
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pyproj
import proplot as pplt
import math

#Accessing the data files for asfs30 data
DF1 = '/Users/saminakashif/Downloads/asfs30.csv'
DF2 = '/Users/saminakashif/Downloads/ASFS30UCB2.csv'
DF3 = '/Users/saminakashif/Downloads/ASFS30UCB3.csv'
DF4 = '/Users/saminakashif/Downloads/ASFS50UCB3.csv'
DF5 = '/Users/saminakashif/Downloads/ASFSTUCB2.csv'

Data1 = pd.read_csv(DF1)
Data2 = pd.read_csv(DF2)
Data3 = pd.read_csv(DF3)
Data4 = pd.read_csv(DF4)
Data5 = pd.read_csv(DF5)

#Filtering the data for just the date and speed data
selected_columns = Data5[['datetime', 'u_wind', 'v_wind', 'speed']]
selected_columns.to_csv('wind_speed.csv', index=False)
speed_data2 = pd.read_csv('wind_speed.csv')

#Calculating the angles of the data
speed_data2['wind_angle'] = np.arctan(speed_data2['u_wind'] / speed_data2['v_wind'])
speed_data2['wind_angle'] = speed_data2['wind_angle'].fillna(0) 

print (speed_data2)

#Creating a plot of the turning angles and how they progressed over the time frame of the data
fig, axs = pplt.subplots(width=7, height=5, nrows=1, spany=False)
for ax, dir in zip(axs, ['wind_angle']):
    ax.plot(speed_data2['datetime'], speed_data2['wind_angle'], label='Estimate')
    ax.format(ylabel='wind angle (radians)', xlabel='date', xrotation=45)
    step = len(speed_data2) // 10
    ax.format(xticks=step)
ax.legend(loc='ll', ncols=1)
