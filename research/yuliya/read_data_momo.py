#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:57:30 2021

@author: marchett
"""

import glob
import numpy as np
import h5py
import netCDF4 as nc4
from scipy.io import netcdf
import matplotlib as mp
from matplotlib import pyplot as plt
import cartopy
import cartopy.crs as ccrs


path = '/Volumes/MLIA_active_data/data_SUDSAQ/MOMO_test/'
model_file = 'outputs/2hr_o3_2005_sfc.nc'


inputs = {}
input_files = glob.glob(path + 'inputs/*2012*')
for d in range(len(input_files)):
    nc_in = netcdf.netcdf_file(input_files[d],'r')
    k = list(nc_in.variables.keys())[-1]

    inputs['lon'] = nc_in.variables['lon'][:]
    inputs['lat'] = nc_in.variables['lat'][:]
    inputs['time'] = nc_in.variables['time'][:]
    inputs[k] = nc_in.variables[k][:]
    

outputs = {}
output_files = glob.glob(path + 'outputs/*_o3_2012*')
for d in range(len(output_files)):
    nc_out = netcdf.netcdf_file(output_files[d],'r')
    k = list(nc_out.variables.keys())[-1]

    outputs['lon'] = nc_out.variables['lon'][:]
    outputs['lat'] = nc_out.variables['lat'][:]
    outputs['time'] = nc_out.variables['time'][:]
    outputs[k] = nc_out.variables[k][:]


x, y = np.array(np.meshgrid(outputs['lon'], outputs['lat']))

plt.figure()
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, np.array(outputs['o3'][0, :, :]))
ax.coastlines()





