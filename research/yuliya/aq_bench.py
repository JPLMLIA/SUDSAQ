#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:39:28 2022

@author: marchett
"""
import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import wget
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

data_output_dir = f'{root_dir}/AQ-Bench/' 
filename = f'AQbench_dataset.csv'
#download from ftp 
ftp_address = 'https://b2share.eudat.eu/api/files/6e1d81b9-670f-4166-8fc2-1592b8adb1fd/'
output_file = f'{data_output_dir}/{filename}' 
wget.download(ftp_address + filename, out = output_file)
    

VARS = ['o3_average_values', 'o3_daytime_avg', 'o3_nighttime_avg', 'o3_median',
        'o3_perc25', 'o3_perc75', 'o3_perc90', 'o3_perc98', 'o3_dma8eu',
        'o3_avgdma8epax', 'o3_drmdmax1h', 'o3_w90', 'o3_aot40', 'o3_nvgt070',
        'o3_nvgt100']
        
NUM_FEATURES = ['climatic_zone', 'lon', 'lat', 'alt',
           'relative_alt', 'water_25km',
           'evergreen_needleleaf_forest_25km', 'evergreen_broadleaf_forest_25km',
           'deciduous_needleleaf_forest_25km', 'deciduous_broadleaf_forest_25km',
           'mixed_forest_25km', 'closed_shrublands_25km', 'open_shrublands_25km',
           'woody_savannas_25km', 'savannas_25km', 'grasslands_25km',
           'permanent_wetlands_25km', 'croplands_25km', 'urban_and_built-up_25km',
           'cropland-natural_vegetation_mosaic_25km', 'snow_and_ice_25km',
           'barren_or_sparsely_vegetated_25km', 'wheat_production',
           'rice_production', 'nox_emissions', 'no2_column', 'population_density',
           'max_population_density_5km', 'max_population_density_25km',
           'nightlight_1km', 'nightlight_5km', 'max_nightlight_25km']

CAT_FEATURES = ['climatic_zone', 'type', 'type_of_area']
OTHER = ['id', 'country', 'htap_region', 'dataset']


data = pd.read_csv(output_file)

#process cat variables
x_cat = np.zeros((data.shape[0], len(CAT_FEATURES)))
for i in range(len(CAT_FEATURES)):
    x = np.array(data[CAT_FEATURES[i]])
    unique_cat = np.unique(x)
    #x = np.zeros(x_cat.shape)
    for j, u in enumerate(unique_cat):
        mask = x== u
        x_cat[mask, i] = j+1

y = np.array(data['o3_avgdma8epax']) 
x = np.column_stack([x_cat, np.array(data[NUM_FEATURES])])
x = np.array(x)

#kfold = GroupKFold(n_splits=len(data_files))
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
for train_index, test_index in kfold.split(x):
    train_x, test_x = x[train_index], x[test_index]
    train_y, test_y = y[train_index], y[test_index]
    train_mask, test_mask = mask[train_index], mask[test_index]

    rf_predictor = RandomForestRegressor()
    rf_predictor.fit(train_x[train_mask],
                     train_y[train_mask])

    out_model = os.path.join(models_month_dir,
                             'rf_%s_cv%d.joblib' % (model_mode, cv_index))
    dump(rf_predictor, out_model)
    print(f'[INFO] Saved trained model: {out_model}')

    pred_y, bias, contribution = ti.predict(rf_predictor, test_x)
    pred_y = pred_y.flatten()


 kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

rf_predictor = RandomForestRegressor()

rf_predictor.fit(train_x, train_y)
    preds_y = rf_predictor.predict(train_x)









    
  
    
    