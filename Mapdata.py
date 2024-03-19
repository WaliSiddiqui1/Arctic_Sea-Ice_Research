#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:07:25 2024

@author: saminakashif
"""

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
from scipy.interpolate import interp1d
import xarray as xr

Df = '/Users/saminakashif/Downloads/ocn_22m_current_preliminary.csv'
data1 = pd.read_csv(Df)

data1['speed'] = (data1['u_ice'] ** 2 + data1['v_ice'] ** 2) ** 0.5
data1 = data1.rename(columns = {'TIME': 'datetime'})
print (data1)

data2 = pd.read_csv('/Users/saminakashif/Downloads/asfs30.csv')
selected_columns = data2[['datetime', 'latitude', 'longitude', 'temp']]
selected_columns.to_csv("temp_space.csv", index=False)
newdata2 = pd.read_csv("temp_space.csv")

merged_data = pd.merge(data1, newdata2, on='datetime', how='inner')

merged_data.to_csv("merged_data.csv", index=False)

Data = pd.read_csv('merged_data.csv')
print (Data)

pplt.rc['reso'] = 'med'

fig, ax = pplt.subplots(proj='ortho', proj_kw={'lon_0': -5, 'lat_0': 75}, width=5)
ax.format(latlim=(70, 85), lonlim=(-25,10),  land=True,
          landcolor='light gray', latlabels=True, lonlabels=True)

depth = xr.open_dataset('/Users/saminakashif/Downloads/interpolated_depth.nc')

cbar = ax.contourf(depth.longitude,
                depth.latitude,
                depth.z, levels=[-4000, -3500, -3000, -2500,
                              -2000, -1500, -1000, -500,
                              -200, -100, -50, 0],
                cmap='blues8_r',
                extend='both')

temp_gradient = ax.scatter(Data.longitude_y, #scatter plot not contour
                            Data.latitude_x,
                            c = Data.temp, #merge operation instead
                            cmap='coolwarm',
                            extend='both')

for buoy in Data.index:  
    ax.plot(Data.loc[buoy, 'longitude_y'], 
            Data.loc[buoy, 'latitude_x'], color='gold', lw=1, alpha=0.75, zorder=2)       

colors = {m: c['color'] for m, c in zip(['May', 'June', 'July', 'August', 'September'],
                                        pplt.Cycle('spectral', N=5))}
        

h = [ax.plot([],[], c=colors[c], marker='o', lw=0, edgecolor='k') for c in colors if c[0] != 'S']
l = [c[0:3] + ' 1st' for c in colors if c[0] != 'S']
ax.legend(h, l, ncols=1, loc='lr', pad=1, alpha=1)

h = [ax.plot([],[], c='light gray', lw=2.5,
            path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()]),
     ax.plot([],[],c='r', lw=2.5), ax.plot([],[], lw=2.5, color='gold')]

ax.colorbar(cbar, label='Depth (m)', loc='b')
ax.colorbar(temp_gradient, label='Temperature (°C)', loc='r')
