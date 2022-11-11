#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:23:48 2022
@author: marchett
@author: kdoerksen - refactored for local use
Runs simple RF model without having to run full
ML pipeline. Good for quick experiments.
"""
import os, glob
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xarray as xr
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    to be -180 to 180
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)

def calc_topten_importance(model, feature_names):
    '''
    Returns top ten features as ranked
    by gini in list
    '''
    importances = model.feature_importances_
    fi_labeled = pd.Series(importances, index=feature_names)
    # sort by highest to lowest
    fi_labeled = fi_labeled.sort_values(ascending=False)
    top_ten = fi_labeled.head(10).index.tolist()
    return top_ten

# Set boundaries to analyze by region
bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80], }

root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'
if not os.path.exists(root_dir):
    root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'

script_home = '/Users/kelseydoerksen/code/suds-air-quality/research/kelsey/run_rf.py'
Logger = logging.getLogger(script_home)
logging.basicConfig(level = logging.INFO)

# Specify Region
for aoi in bbox_dict.keys():
    print('Running RF model using training data from region: {}'.format(aoi))
    region = aoi
    bbox = bbox_dict[region]

    # Set root directory to grab things from
    Logger.info('Setting directories')

    month = 'jul'
    years = [2011, 2012, 2013, 2014, 2015]
    months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']

    # set the directory for that month
    models_dir = f'{root_dir}/models/2011-2015/bias-8hour/'
    # set plot directory
    summaries_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/combined_data/'
    plots_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/plots/'
    # create directories if they don't exist
    dirs = [models_dir, summaries_dir, plots_dir]
    for direct in dirs:
        if not os.path.exists(direct):
            os.makedirs(direct)

    month = 'jul'
    Logger.debug('Grabbing data from month {}'.format(month))
    training_x = np.hstack(glob.glob(f'{root_dir}/{month}/*/train.data.nc'))
    testing_x = np.hstack(glob.glob(f'{root_dir}/{month}/*/test.data.nc'))

    # If [#] at the end, grabbing only one file to work with
    training_y = glob.glob(f'{root_dir}/{month}/*/train.target.nc')[0]
    testing_y = glob.glob(f'{root_dir}/{month}/*/test.target.nc')[0]

    # Read in toar training target (predictand)
    toar_ds = xr.open_dataset(training_y)

    # Filter data by lat-lon bounds specific in bbox
    toar_ds = format_lon(toar_ds)
    toar_ds = toar_ds.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))

    y = toar_ds['target'].values
    # Remove NaNs
    mask_training = ~np.isnan(y)
    y = y[mask_training]

    # Read in toar testing target
    toar_test = xr.open_dataset(testing_y)
    y_test = toar_test['target'].values
    # Remove Nans
    mask_testing = ~np.isnan(y_test)
    y_test = y_test[mask_testing]

    # Read in training data
    # If [#] at the end, grabbing only one file to work with
    data = xr.open_dataset(training_x[0])
    # Filter data by lat-lon bounds specific in bbox
    data = format_lon(data)
    data = data.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))

    var_names = list(data.keys())
    X = []
    for j in tqdm(range(len(var_names))):
        x = data[var_names[0]].values[mask_training]
        X.append(x)
    X = np.column_stack(X)

    data_test = xr.open_dataset(testing_x[0])
    X_test = []
    for j in tqdm(range(len(var_names))):
        x = data_test[var_names[j]].values[mask_testing]
        X_test.append(x)
    X_test = np.column_stack(X_test)

    if len(X) < 1:
        print('There is no training data for region: {}, skipping'.format(aoi))
        continue

    Logger.info('Running Random Forest')
    rf = RandomForestRegressor(n_estimators=20,
                               max_features=int(len(var_names) * 0.3),
                               random_state=6789)
    rf.fit(X, y)
    yhat = rf.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, yhat))
    print('mse is {}'.format(mse))
    print('the top ten features are {}'.format(calc_topten_importance(rf, var_names)))

    '''
    Logger.info('Plotting y_test vs y_hat')
    plt.figure()
    plt.plot(y_test, yhat, '.')
    plt.show()
    '''