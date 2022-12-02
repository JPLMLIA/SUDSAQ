#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kdoerksen - refactored from marchett for local use
Runs simple RF model without having to run full
ML pipeline. Good for quick experiments.
"""

import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
import xarray as xr
from tqdm import tqdm
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import seaborn as sns

parser = argparse.ArgumentParser(description='Local RF Run')
parser.add_argument("--analysis_year", help="Specify year to use for test, all other years used for training."
                                        "Must be between 2011-2015 (inclusive)")
parser.add_argument("--region", help="Boundary region on Earth to take data. Must be one of: "
                                     "globe, europe, asia, australia, north_america, west_europe, "
                                     "east_europe, west_na, east_na.")
parser.add_argument("--month", help="Month of analysis. Currently only support jan, jul")
args = parser.parse_args()

script_home = '/Users/kelseydoerksen/code/suds-air-quality/research/kelsey/run_rf.py'
Logger = logging.getLogger(script_home)
logging.basicConfig(level = logging.INFO)

# Set root directory to grab things from
Logger.info('Setting directories')
root_dir = '/Users/kelseydoerksen/exp_runs/rf/'
gee_dir = '/Users/kelseydoerksen/gee'

def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    to be -180 to 180
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)

def format_gee(gee, train_x):
    '''
    Formatting gee added data to match
    training data dim order, time
    '''
    # add time dims to match momo (fake, values are constant)
    gee = gee.expand_dims(time=train_x['time'])
    gee_t = gee.transpose('lat','lon','time')

    # match lon to momo
    gee_t.coords['lon'] = gee_t.coords['lon'] + 180

    # fill nans with 0 as category for missing data
    gee_final = gee_t.fillna(0)

    return gee_final

def generate_X_and_y(X_train, y_train, modis, fire, pop):
    '''
    Generates X for input, y for target
    for RF fit
    '''
    # Read in toar training target (predictand)
    toar_ds = xr.open_dataset(y_train)

    # Filter data by lat-lon bounds specific in bbox
    toar_ds = format_lon(toar_ds)
    toar_ds = toar_ds.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))

    y_t = toar_ds['target'].values

    # Remove NaNs
    mask_training = ~np.isnan(y_t)
    y_t = y_t[mask_training]

    # Read in training data
    data = xr.open_dataset(X_train)

    # Dealing with gee
    modis_data = xr.open_dataset(modis)
    fire_data = xr.open_dataset(fire)
    pop_data = xr.open_dataset(pop)
    combined_data = data.merge(modis_data).merge(fire_data).merge(pop_data)

    # Filter data by lat-lon bounds specific in bbox
    data = format_lon(combined_data)
    data = data.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))

    variable_names = list(data.keys())
    x = []
    for j in tqdm(range(len(variable_names))):
        x_data = data[variable_names[j]].values[mask_training]
        x.append(x_data)
    x_t = np.column_stack(x)

    return x_t, y_t

def calc_importances(model, feature_names):
    '''
    Calculate feature importances, save as txt
    and plot
    '''
    importances = model.feature_importances_
    fi_labeled = pd.Series(importances, index=feature_names)
    # sort by highest to lowest
    fi_labeled = fi_labeled.sort_values(ascending=False)
    print(fi_labeled)
    fi_labeled.plot.bar()
    plt.title('Feature Importance')
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.show()
    plt.savefig('testing.png')

def truth_vs_predicted(target, predict, label=None, save=None):
    """
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Retrieve the limits and expand them by 5% so everything fits into a square grid
    limits = min([target.min(), predict.min()]), max([target.max(), predict.max()])
    limits = limits[0] - np.abs(limits[0] * .05), limits[1] + np.abs(limits[1] * .05)
    ax.set_ylim(limits)
    ax.set_xlim(limits)

    # Create the horizontal line for reference
    ax.plot((limits[0], limits[1]), (limits[0], limits[1]), '--', color='r')

    # Create the density values
    kernel = stats.gaussian_kde([target, predict])
    density = kernel([target, predict])

    plot = ax.scatter(target, predict, c=density, cmap='viridis', label=label, s=5)

    # Create the colorbar without ticks
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_ticks([])

    # Set labels
    cbar.set_label('Density')
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Truth vs Predicted')

    if label:
        legend = ax.legend(handlelength=0, handletextpad=0, loc='upper left', prop={'family': 'monospace'})
        legend.legendHandles[0].set_visible(False)

    plt.axis('equal')
    plt.tight_layout()
    if save:
        Logger.info(f'Saving truth_vs_predicted plot to {save}')
        plt.savefig(save)
    plt.show()

def plot_true_vs_predicted_map():
    '''
    Generates map plot of true vs predicted bias
    '''


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

aoi = args.region
print('Running RF model using training data from region: {}'.format(aoi))
bbox = bbox_dict[aoi]

analysis_year = int(args.analysis_year)
month = args.month

Logger.debug('Grabbing data from month {}'.format(month))

# Dealing with training data
x_train = f'{root_dir}/{month}/{analysis_year}/train.data.nc'
y_train = f'{root_dir}/{month}/{analysis_year}/train.target.nc'

modis_train = f'{root_dir}/{month}/{analysis_year}/train.modis.nc'
fire_train = f'{root_dir}/{month}/{analysis_year}/train.fire.nc'
pop_train = f'{root_dir}/{month}/{analysis_year}/train.pop.nc'

X, y = generate_X_and_y(x_train, y_train, modis_train, fire_train, pop_train)

# --- Dealing with testing data ---
testing_x = f'{root_dir}/{month}/{analysis_year}/test.data.nc'
testing_y = f'{root_dir}/{month}/{analysis_year}/test.target.nc'
modis_test = f'{root_dir}/{month}/{analysis_year}/test.modis.nc'
fire_test = f'{root_dir}/{month}/{analysis_year}/test.fire.nc'
pop_test = f'{root_dir}/{month}/{analysis_year}/test.pop.nc'

# Read in toar testing target
toar_test = xr.open_dataset(testing_y)
y_test = toar_test['target'].values
# Remove Nans
mask_testing = ~np.isnan(y_test)
y_test = y_test[mask_testing]

data_test = xr.open_dataset(testing_x)
modis_ds = xr.open_dataset(modis_test)
fire_ds = xr.open_dataset(fire_test)
pop_ds = xr.open_dataset(pop_test)

# Read in gee datasets to add to training data
combined_test = data_test.merge(modis_ds).merge(fire_ds).merge(pop_ds)


var_names = list(combined_test.keys())
X_test = []
for j in tqdm(range(len(var_names))):
    x = combined_test[var_names[j]].values[mask_testing]
    X_test.append(x)
X_test = np.column_stack(X_test)

Logger.info('Running Random Forest')
rf = RandomForestRegressor(n_estimators=20,
                           max_features=int(len(var_names) * 0.3),
                           random_state=6789,
                           verbose=1)
rf.fit(X, y)
yhat = rf.predict(X_test)

# Calculate rmse
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
print('rmse is {}'.format(rmse))

# Calculate mape
mape = mean_absolute_percentage_error(y_test, yhat)
print('mean absolute percentage error is: {}'.format(mape))

# Calculate r correlation value
r = pearsonr(y_test, yhat)[0]
print("r correlation is: {}".format(r))

# Calculate r2 score
r2 = r2_score(y_test,yhat)
print('r2 score is: {}'.format(r2))

# Calculate, plot, and save importances
calc_importances(rf, var_names)

# Plot true vs predicted
Logger.info('Plotting y_test vs y_hat')
truth_vs_predicted(y_test, yhat)