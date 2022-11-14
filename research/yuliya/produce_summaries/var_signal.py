#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:33:12 2022

@author: marchett
"""
import os, glob
import sys
import json
import numpy as np
import h5py
import xarray as xr
from scipy.io import netcdf
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
import seaborn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import summary_plots as plots
import read_output as read
from sklearn.preprocessing import StandardScaler




root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

#set plot directory
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/' 
# plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
# # create one if it's not there
# if not os.path.exists(plots_dir):
#     os.makedirs(plots_dir)
bbox = plots.bbox_dict['globe']
if months == 'all':
    months = plots.MONTHS

# make the full correlation matrix
exclude = ['momo.hno3', 'momo.oh', 'momo.pan', 'momo.q2',
           'momo.sens', 'momo.so2', 'momo.T2', 'momo.taugxs',
           'momo.taugys', 'momo.taux', 'momo.tauy', 'momo.twpc',
           'momo.2dsfc.CFC11', 'momo.2dsfc.CFC113', 'momo.2dsfc.CFC12',
           'momo.ch2o', 'momo.cumf0', 'momo.2dsfc.dms']

#run correlations for all months
month = 'jul'    
    
data_file = f'{summaries_dir}/{month}/test.data.mean.nc'
print(data_file)
if not os.path.isfile(data_file):
    continue
 
data = xr.open_dataset(data_file)
data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
data = data.sortby(data.lon)
var_names = list(data.keys())
    
    
    
    
    