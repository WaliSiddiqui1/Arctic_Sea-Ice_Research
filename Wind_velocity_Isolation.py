#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:14:37 2024

@author: saminakashif
"""


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pyproj
import proplot as pplt
import math

#Filtering the data for the date and speed
file = '/Users/saminakashif/Downloads/ASFS30UCB2.csv'
data = pd.read_csv(file)
selected_columns = data[['datetime', 'u_wind', 'v_wind', 'speed']]
selected_columns.to_csv('wind_speed.csv', index=False)
speed_data = pd.read_csv('wind_speed.csv')

#Calcuating the speed of the wind and inputting into the dataframe
speed_data['wind_velocity'] = np.sqrt(speed_data['u_wind'] ** 2 + speed_data['v_wind'] ** 2)
speed_data['wind_velocity'] = speed_data['wind_velocity'].fillna(0)

#Calculating the angle of the data and putting it into the dataframe
speed_data['wind_angle'] = np.arcsin(speed_data['u_wind'] / speed_data['v_wind'])
speed_data['wind_angle'] = speed_data['wind_angle'].fillna(0) 
speed_data['sin_angle'] = np.sin(speed_data['wind_angle'])
speed_data['sin_angle'] = speed_data['sin_angle'].fillna(0)

speed_data.to_csv('wind_speed.csv', index=False)

#Alpha value for the ice velocity calculation
alpha = 0.015

# Converting velocity into a 1D complex number shortens the number of steps
theta = speed_data['wind_angle']
Uwind = speed_data['wind_velocity']
Uice = alpha*np.exp(-1j*theta)*Uwind

# Convert back into real numbers
speed_data['real_speed'] = np.real(Uice)
speed_data['imag_speed'] = np.imag(Uice)

# residual
speed_data['error'] =  speed_data['speed'] - speed_data['real_speed']
R = speed_data['error']
speed_data.to_csv('wind_speed.csv', index=False)

print(speed_data)

#Creating graph that plots the datetime against the calculated velocity to show the progression over time
fig, axs = pplt.subplots(width=7, height=5, nrows=1, spany=False)
for ax, dir in zip(axs, ['real_speed', 'speed']):
    ax.plot(speed_data['datetime'], speed_data['real_speed'], label='Estimate')
    ax.plot(speed_data['datetime'], speed_data['speed'], label='Ice Velocity')
    ax.format(ylabel=dir+'-component of ice velocity (m/s)', xlabel='date', xrotation=45)
    step = len(speed_data) // 10
    ax.format(xticks=step)
ax.legend(loc='ll', ncols=1)

#calc average residual/difference and percent error

print(f'Average residual is {round(R.mean(), 5)}')

percent_err = (R.mean()/speed_data['speed'].mean()) * 100

print(f'percent error is {round(percent_err, 5)}%')




