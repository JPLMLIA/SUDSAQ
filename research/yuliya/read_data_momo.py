#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:57:30 2021

@author: marchett
"""


import numpy as np
import h5py
import netCDF4 as nc4
import matplotlib as mp
from matplotlib import pyplot as plt


path = '/Volumes/MLIA_active_data/data_SUDSAQ/test_data/'
model_file = 'outputs/2hr_o3_2019_sfc.nc'
input_file = 'inputs/2hr_t_2019_sfc.nc'

nc_out = nc4.Dataset(path + model_file, 'r')
nc_in = nc4.Dataset(path + input_file, 'r')

#check variables
nc_out.variables.keys()

#time dim 4380
time = nc_out['time'][:]
#space dim 320 x 160
lon = nc_out['lon'][:]
lat = nc_out['lat'][:]
ozone = nc_out['o3'][:]


x, y = np.array(np.meshgrid(lon, lat))

plt.figure()
plt.pcolor(x, y, np.array(ozone[0, :, :]))


