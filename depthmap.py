#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:48:05 2024

@author: saminakashif
"""

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
from scipy.interpolate import interp1d
import xarray as xr

df1 = '/Users/saminakashif/Downloads/2019T66.csv'
df2 = '/Users/saminakashif/Downloads/2019O1.csv'
df3 = '/Users/saminakashif/Downloads/2019O3.csv'
df4 = '/Users/saminakashif/Downloads/2019O4.csv'
df5 = '/Users/saminakashif/Downloads/2019O5.csv'
df6 = '/Users/saminakashif/Downloads/2019O6.csv'

data = pd.read_csv(df1)
data2 = pd.read_csv(df2)
data3 = pd.read_csv(df3)
data4 = pd.read_csv(df4)
data5 = pd.read_csv(df5)
data6 = pd.read_csv(df6)

#Filtering the columns to fall between the dates of June 133th to July 30th in 2020
selected_columns = data.loc[(data['datetime'] >= '2020-06-13 00:00:00') & 
                            (data['datetime'] <= '2020-07-30 00:00:00')]
selected_columns.to_csv('data.csv', index=False)
data = pd.read_csv('data.csv')

#Filtering the column data for their date, depth, longitude, and latitude
selected_columns1 = data[['datetime','depth', 'longitude', 'latitude']]
selected_columns1.to_csv('data1.csv', index=False)
data1= pd.read_csv('data1.csv')

pplt.rc['reso'] = 'med'

#Configuring the map of the artic surrounding the buoy tracks
fig, ax = pplt.subplots(proj='ortho', proj_kw={'lon_0': -5, 'lat_0': 75}, width=5)
ax.format(latlim=(70, 85), lonlim=(-25,10),  land=True,
          landcolor='light gray', latlabels=True, lonlabels=True)


depth = xr.open_dataset('/Users/saminakashif/Downloads/interpolated_depth.nc')

#Zooming the map in on the area surrounding the buoy tracks
min_lon = data1.longitude.min()
max_lon = data1.longitude.max()
min_lat = data1.latitude.min()
max_lat = data1.latitude.max()
ax.format(latlim=(min_lat, max_lat), lonlim=(min_lon, max_lon))

#Contouring the depths along their corresponding longitude and latitude within the map
cbar = ax.contourf(depth.longitude,
                depth.latitude,
                depth.z, levels=[-4500, -4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                cmap='blues8_r',
                extend='both')
track = ax.scatter(data1.longitude, #scatter plot not contour
                            data1.latitude,
                            c = data1.depth, levels=[-4500, -4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                            cmap='coolwarm',
                            extend='both')

#Filtering the second data file
selected_columns2 = data2.loc[(data2['datetime'] >= '2020-06-13 00:00:00') & 
                            (data2['datetime'] <= '2020-07-30 00:00:00')]
selected_columns2.to_csv('data2.csv', index=False)
data2 = pd.read_csv('data2.csv')
selected_columns3 = data2[['datetime','depth', 'longitude', 'latitude']]
selected_columns3.to_csv('data2.csv', index=False)
data2 = pd.read_csv('data2.csv')

#Putting the tracks of data2's depth along the map
track2 = ax.scatter(data2.longitude, #scatter plot not contour
                            data2.latitude,
                            c = data2.depth, levels=[-4500, -4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                            cmap='yellow',
                            extend='both')



selected_columns4 = data3.loc[(data3['datetime'] >= '2020-06-13 00:00:00') & 
                            (data3['datetime'] <= '2020-07-30 00:00:00')]

selected_columns4.to_csv('data3.csv', index=False)
data3 = pd.read_csv('data3.csv')

selected_columns5 = data3[['datetime','depth', 'longitude', 'latitude']]

selected_columns5.to_csv('data3.csv', index=False)

data3 = pd.read_csv('data3.csv')

track3 = ax.scatter(data3.longitude, #scatter plot not contour
                            data3.latitude,
                            c = data3.depth, levels=[-4500, -4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                            cmap='pink',
                            extend='both')

selected_columns6 = data4.loc[(data4['datetime'] >= '2020-06-13 00:00:00') & 
                            (data4['datetime'] <= '2020-07-30 00:00:00')]

selected_columns6.to_csv('data4.csv', index=False)
data4 = pd.read_csv('data4.csv')

selected_columns7 = data4[['datetime','depth', 'longitude', 'latitude']]

selected_columns7.to_csv('data4.csv', index=False)

data4 = pd.read_csv('data4.csv')

track4 = ax.scatter(data4.longitude, #scatter plot not contour
                            data4.latitude,
                            c = data4.depth, levels=[-4500, -4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                            cmap='white',
                            extend='both')

#Marking spots of the yermak plateu transition based on depth changes
for i in range (1, len(data1[['depth']])):
    data1['difference'] = data1['depth'].iloc[i] - data1['depth'].iloc[i-1]
    if data1['depth'].iloc[i] < -3000 and sum(data1['difference'].iloc[i::i+5]) >= abs(1500):
        print (data1.iloc[i])
        marker1 = ax.scatter(data1.longitude[i], 
                                    data1.latitude[i],
                                    c = data1.depth[i], levels=[-4000, -3500, -3000, -2500,
                                      -2000, -1500, -1000, -500,
                                      -200, -100, -50, 0],
                                    cmap='black',
                                    extend='both')
    else:
        pass   

#Marking the area of the Fram Strait transition
for j in range (1, len(data1[['depth']])):
    data1['difference'] = data1['depth'].iloc[j] - data1['depth'].iloc[j-1]
    if data1['depth'].iloc[j] == -1788.6066715429915:
        print (data1.iloc[j])
        marker2 = ax.scatter(data1.longitude[j], #scatter plot not contour
                                    data1.latitude[j],
                                    c = data1.depth[j], levels=[-4000, -3500, -3000, -2500,
                                      -2000, -1500, -1000, -500,
                                      -200, -100, -50, 0],
                                    cmap='green',
                                    extend='both')
print (data1)


for buoy in data1.index:  
    ax.plot(data1.loc[buoy, 'longitude'], 
            data1.loc[buoy, 'latitude'], color='gold', lw=1, alpha=0.75, zorder=2)       

colors = {m: c['color'] for m, c in zip(['June', 'July'],
                                        pplt.Cycle('spectral', N=5))}
        
#Creating a legend of the map data
h = [ax.plot([],[], c=colors[c], marker='o', lw=0, edgecolor='k') for c in colors if c[0] != 'S']
l = [c[0:3] + ' 1st' for c in colors if c[0] != 'S']
ax.legend(h, l, ncols=1, loc='lr', pad=1, alpha=1)

h = [ax.plot([],[], c='light gray', lw=2.5,
            path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()]),
     ax.plot([],[],c='r', lw=2.5), ax.plot([],[], lw=2.5, color='gold')]

#Creating side bar color legends of the tracks on the map
ax.colorbar(cbar, label='Depth (m)', loc='b')
ax.colorbar(track, label='depth along track', loc='l')
ax.colorbar(marker1, label = 'Yermak Plateau', loc ='r')
ax.colorbar(marker2, label = 'Fram Strait', loc = 'r')



