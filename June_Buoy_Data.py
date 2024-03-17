#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:44:31 2024

@author: saminakashif
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pyproj
import xarray as xr

Buoy_data = pd.read_csv(r"/Users/saminakashif/Downloads/DN_buoy_list_v2.csv")
print(Buoy_data.head())
Temp_data = pd.read_csv(r'/Users/saminakashif/Downloads/L2archive.csv')
print (Temp_data)
columns_to_stay = [4, 6, 9, 10]
filtered_temp2 = Temp_data.iloc[1497:1592, columns_to_stay]

print(filtered_temp2)
pr = xr.open_dataset('/Users/saminakashif/Downloads/mosmet.asfs30.level3.4.1min.20200214.000000.nc')

file_mapping = {
    "2020E1.csv": "June_data_E1",
    "2020E2.csv": "June_data_E2",
    "2020T79.csv": "June_data_T79",
    "2020T77.csv": "June_data_T77",
    "2020T76.csv": "June_data_T76",
    "2020T75.csv": "June_data_T75",
    "2020T74.csv": "June_data_T74",
    "2020T73.csv": "June_data_T73",
    "2020T61.csv": "June_data_T61",
    "2020S99.csv": "June_data_S99",
    "2020S97.csv": "June_data_S97",
    "2020R12.csv": "June_data_R12",
    "2020R11.csv": "June_data_R11",
    "2020R10.csv": "June_data_R10",
    "2020P228.csv": "June_data_P228",
    "2020P227.csv": "June_data_P227",
    "2020P226.csv": "June_data_P226",
    "2020P225.csv": "June_data_P225",
    "2020P223.csv": "June_data_P223",
    "2020P220.csv": "June_data_P220",
    "2020P219.csv": "June_data_P219",
    "2020P218.csv": "June_data_P218",
    "2020P217.csv": "June_data_P217",
    "2020P216.csv": "June_data_P216",
    "2020P215.csv": "June_data_P215",
    "2020P213.csv": "June_data_P213",
    "2020P211.csv": "June_data_P211",
    "2020P210.csv": "June_data_P210",
    "2020P186.csv": "June_data_P186",
    "2020P160.csv": "June_data_P160",
    "2020E3.csv": "June_data_E3"
}

base_path = "/Users/saminakashif/Downloads/"

for file_name, new_name in file_mapping.items():
    file_path = base_path + file_name
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    df.index = pd.to_datetime(df.index)
    filtered_data = df.loc['2020-06-15':'2020-06-30']
    sorted_data = filtered_data.sort_index()
    globals()[new_name] = sorted_data  
    sorted_data.to_csv(new_name + '_sorted.csv')

print(June_data_E1)

def check_positions(buoy_df, pairs_only=False,
                   latname='latitude', lonname='longitude'):
    """Looks for duplicated or nonphysical position data. Defaults to masking any 
    data with exact matches in latitude or longitude. Setting pairs_only to false 
    restricts the check to only flag where both longitude and latitude are repeated
    as a pair.
    """

    lats = buoy_df[latname].round(10)
    lons = buoy_df[lonname].round(10)
    
    invalid_lats = np.abs(lats) > 90
    if np.any(lons < 0):
        invalid_lons = np.abs(lons) > 180
    else:
        invalid_lons = lons > 360
        
    invalid = invalid_lats | invalid_lons
    
    repeated = lats.duplicated(keep='first') | lons.duplicated(keep='first')
    
    duplicated = pd.Series([(x, y) for x, y in zip(lons, lats)],
                                  index=buoy_df.index).duplicated(keep='first')
    
    if pairs_only:
        return duplicated | invalid
    
    else:
         return repeated | duplicated | invalid


def check_dates(buoy_df, precision='1min', date_col=None):
    """Check if there are reversals in the time or duplicated dates. Optional: check
    whether data are isolated in time based on specified search windows and the threshold
    for the number of buoys within the search windows. Dates are rounded to <precision>,
    so in some cases separate readings that are very close in time will be flagged
    as duplicates. Assumes date_col is in a format readable by pandas to_datetime.
    """

    if date_col is None:
        date_values = buoy_df.index.values
        date = pd.Series(pd.to_datetime(date_values).round(precision),
                     index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df[date_col]).round(precision)
    duplicated_times = date.duplicated(keep='first')
    
    time_till_next = date.shift(-1) - date
    time_since_last = date - date.shift(1)

    negative_timestep = time_since_last.dt.total_seconds() < 0

    return negative_timestep | duplicated_times

def check_gaps(buoy_df, threshold_gap='4H', threshold_segment=12, date_col=None):
    """Segments the data based on a threshold of <threshold_gap>. Segments shorter
    than <threshold_segment> are flagged."""
    
    if date_col is None:
        date_values = buoy_df.index.values
        date = pd.Series(pd.to_datetime(date_values),
                     index=buoy_df.index)
    else:
        date = pd.to_datetime(buoy_df[date_col])
    
    time_till_next = date.shift(-1) - date
    segment = pd.Series(0, index=buoy_df.index)
    counter = 0
    tg = pd.to_timedelta(threshold_gap)
    for t in segment.index:
        segment.loc[t] = counter
        if time_till_next[t] > tg:
            counter += 1
    
    # apply_filter
    new = buoy_df.groupby(segment).filter(lambda x: len(x) > threshold_segment).index
    flag = pd.Series(True, index=buoy_df.index)
    flag.loc[new] = False
    return flag

    

def compute_velocity(buoy_df, date_index=True, rotate_uv=False, method='c'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stereographic x/y prior
    to calculating velocity. Rotate_uv moves the velocity into east/west. Velocity
    calculations are done on the provided time index. Results will not necessarily 
    be reliable if the time index is irregular. With centered differences, values
    near endpoints are calculated as forward or backward differences.
    
    Options for method
    forward (f): forward difference, one time step
    backward (b): backward difference, one time step
    centered (c): 3-point centered difference
    forward_backward (fb): minimum of the forward and backward differences
    """
    buoy_df = buoy_df.copy()
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date)
        
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    
    if 'x' not in buoy_df.columns:
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC North Polar Stereographic
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df['x'] = x
        buoy_df['y'] = y
    
    if method in ['f', 'forward']:
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'])/dt
        dydt = (buoy_df['y'].shift(-1) - buoy_df['y'])/dt

    elif method in ['b', 'backward']:
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df['x'] - buoy_df['x'].shift(1))/dt
        dydt = (buoy_df['y'] - buoy_df['y'].shift(1))/dt

    elif method in ['c', 'fb', 'centered', 'forward_backward']:
        fwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='forward')
        bwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='backward')

        fwd_dxdt, fwd_dydt = fwd_df['u'], fwd_df['v']
        bwd_dxdt, bwd_dydt = bwd_df['u'], bwd_df['v']
        
        if method in ['c', 'centered']:
            dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
            dxdt = (buoy_df['x'].shift(-1) - buoy_df['x'].shift(1))/dt
            dydt = (buoy_df['y'].shift(-1) - buoy_df['y'].shift(1))/dt
        else:
            dxdt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dxdt, 'b':bwd_dxdt})).min(axis=1)
            dydt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dydt, 'b':bwd_dydt})).min(axis=1)

        dxdt.loc[fwd_endpoint] = fwd_dxdt.loc[fwd_endpoint]
        dxdt.loc[bwd_endpoint] = bwd_dxdt.loc[bwd_endpoint]
        dydt.loc[fwd_endpoint] = fwd_dydt.loc[fwd_endpoint]
        dydt.loc[bwd_endpoint] = bwd_dydt.loc[bwd_endpoint]
    
    if rotate_uv:
        # Unit vectors
        buoy_df['Nx'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['x']
        buoy_df['Ny'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ex'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * -buoy_df['y']
        buoy_df['Ey'] = 1/np.sqrt(buoy_df['x']**2 + buoy_df['y']**2) * buoy_df['x']

        buoy_df['u'] = buoy_df['Ex'] * dxdt + buoy_df['Ey'] * dydt
        buoy_df['v'] = buoy_df['Nx'] * dxdt + buoy_df['Ny'] * dydt

        # Calculate angle, then change to 360
        heading = np.degrees(np.angle(buoy_df.u.values + 1j*buoy_df.v.values))
        heading = (heading + 360) % 360
        
        # Shift to direction from north instead of direction from east
        heading = 90 - heading
        heading = (heading + 360) % 360
        buoy_df['bearing'] = heading
        buoy_df['speed'] = np.sqrt(buoy_df['u']**2 + buoy_df['v']**2)
        buoy_df.drop(['Nx', 'Ny', 'Ex', 'Ey'], axis=1, inplace=True)
        
    else:
        buoy_df['u'] = dxdt
        buoy_df['v'] = dydt            
        buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)    

    return buoy_df

check_positions(June_data_E1)
check_dates(June_data_E1)
check_gaps(June_data_E1)
print(compute_velocity(June_data_E1))

def calculate_flux(sea_ice_thickness, sea_ice_temperature, air_temperature, wind_speed):
  """Calculates the flux through sea ice."

  Args:
    sea_ice_thickness: The thickness of the sea ice in meters.
    sea_ice_temperature: The temperature of the sea ice in degrees Celsius.
    air_temperature: The temperature of the air in degrees Celsius.
    wind_speed: The wind speed in meters per second.

  Returns:
    The flux through the sea ice in watts per square meter."""
  
    #Fw = 1/Δt * (Qf + Qs + QL)
  thermal_conductivity = 2.22 * np.power(sea_ice_temperature, -1.5)
  temperature_gradient = sea_ice_temperature - air_temperature
  flux = thermal_conductivity * temperature_gradient * wind_speed

  return flux
filtered_Temp = filtered_temp2.apply(pd.to_numeric, errors='coerce')
print (filtered_temp2)
sorted_data = sorted_data[:-(384-(384-95))]
sorted_data['flux'] = calculate_flux(pd.to_numeric(filtered_temp2.iloc[4]), pd.to_numeric(filtered_temp2.iloc[2]), pd.to_numeric(filtered_temp2.iloc[1]), pd.to_numeric(sorted_data['u_wind']))
average_flux = sorted_data['flux'].mean()
average_fluxes = {}
average_fluxes['average_fluxes'] = average_flux
sorted_data['flux'].to_csv('average_fluxes' + '_of_buoys.csv', header = True)
overall_average_flux = average_flux
average_flux_df = pd.read_csv('average_fluxes_of_buoys.csv')
print(average_flux_df)
