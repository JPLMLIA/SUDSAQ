"""
@author: kdoerksen
Runs simple RF model without having to run full
ML pipeline. Good for quick experiments.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
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
from joblib import dump
from sudsaq.utils import align_print
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='Local RF Run')
parser.add_argument("--analysis_year", help="Specify year to use for test, all other years used for training."
                                        "Must be between 2011-2015 (inclusive)")
parser.add_argument("--region", help="Boundary region on Earth to take data. Must be one of: "
                                     "globe, europe, asia, australia, north_america, west_europe, "
                                     "east_europe, west_na, east_na.")
parser.add_argument("--month", help="Month of analysis. Currently only support jan, jul")
parser.add_argument("--add_gee", help="Specify whether to incorporate gee datasets with y")
parser.add_argument("--parameter_tuning", help="Specify whether to incorporate param tuning with y")

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
    OLD, don't need anymore
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

def generate_X_and_y(X_train, y_train, modis, fire, pop, gee):
    '''
    Generates X for input, y for target
    for RF fit
    '''
    # Read in toar training target (predictand)
    toar_ds = xr.open_dataset(y_train)

    # Filter data by lat-lon bounds specific in bbox
    toar_ds = format_lon(toar_ds)
    y_toar = toar_ds.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))
    y_t = y_toar['target'].values

    # Remove NaNs
    mask_training = ~np.isnan(y_t)
    y_t = y_t[mask_training]

    # Read in training data
    data = xr.open_dataset(X_train)
    data = format_lon(data)

    if gee:
        # Dealing with gee
        modis_data = xr.open_dataset(modis)
        #fire_data = xr.open_dataset(fire)
        pop_data = xr.open_dataset(pop)
        #combined_data = data.merge(modis_data).merge(fire_data).merge(pop_data)
        combined_data = data.merge(modis_data).merge(pop_data)
    else:
        combined_data = data

    # Filter data by lat-lon bounds specific in bbox
    combined_data = combined_data.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))

    variable_names = list(combined_data.keys())
    x = []
    for j in tqdm(range(len(variable_names))):
        x_data = combined_data[variable_names[j]].values[mask_training]
        x.append(x_data)
    x_t = np.column_stack(x)

    return x_t, y_t

def calc_importances(model, feature_names, dir):
    '''
    Calculate feature importances, save as txt
    and plot
    '''
    importances = model.feature_importances_
    stddev = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    df = pd.DataFrame(np.array([importances, stddev]), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    with open('{}/importances.txt'.format(dir), 'w') as file:
        file.write('\n\nFeature Importance:\n')
        file.write('\n'.join(strings))

    return df

def calc_perm_importance(model, data, target, feature_names, dir):
    permimp = permutation_importance(model, data, target, random_state=0)
    # Only want the summaries, remove the importances array
    del permimp['importances']

    # Convert to a DataFrame and sort by importance value
    df = pd.DataFrame(permimp.values(), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    with open('{}/perm_importances.txt'.format(dir), 'w') as file:
        file.write('\n\nPermutation Feature Importance:\n')
        file.write('\n'.join(strings))

    return df

def plot_importances(imp, perm, dir, month, year, region):
    '''
    Generates bar plot for FI with perm
    '''

    # Normalize first
    imp = imp / imp.max(axis=1).importance
    perm = perm / perm.max(axis=1).importance

    # Plot only top 15 features
    imp_top = imp[imp.columns[:20]]
    perm_top = perm[perm.columns[:20]]
    X_axis = np.arange(20)
    plt.bar(X_axis - 0.2, imp_top.loc['importance'].values, yerr=imp_top.loc['stddev'].values,
            width=0.4, label='Importance')
    plt.bar(X_axis + 0.2, perm_top.loc['importance'].values, yerr=perm_top.loc['stddev'].values,
            width=0.4, label='Permutation Importance')
    plt.xticks(X_axis, imp_top, rotation=90)
    plt.title('Feature Importances for Testing set {} {} {}'.format(month, year, region))
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/importances.png'.format(dir))
    plt.show()

def truth_vs_predicted(target, predict, dir, region):
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

    plot = ax.scatter(target, predict, c=density, cmap='viridis', s=5)

    # Create the colorbar without ticks
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_ticks([])

    # Set labels
    cbar.set_label('Density')
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Truth vs Predicted for {}'.format(region))

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('{}/truth_vs_pred.png'.format(dir))
    plt.show()

def plot_histogram(target, pred, dir, region):
    '''
    Plot histogram of true vs predicted
    '''
    bins = np.linspace(-60, 60, 100)
    plt.hist(target, bins, histtype='step', label=['target'])
    plt.hist(pred, bins, histtype='step', label=['prediction'])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('bias')
    plt.ylabel('count')
    plt.title('Truth vs Predicted Histogram for {}'.format(region))
    plt.savefig('{}/truth_vs_pred_hist.png'.format(dir))
    plt.show()

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
add_gee = args.add_gee
param_tuning = args.parameter_tuning

if not add_gee:
    results_dir = '/Users/kelseydoerksen/exp_runs/rf/{}/all_gee_added/{}/results_no_gee/{}/test'.\
        format(month, analysis_year, aoi)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
else:
    results_dir = '/Users/kelseydoerksen/exp_runs/rf/{}/all_gee_added/{}/results/{}/test'.\
        format(month, analysis_year, aoi)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

Logger.debug('Grabbing data from month {}'.format(month))

# --- Below is a total mess and I need to clean it up but works for now ---

# Dealing with training data
x_train = f'{root_dir}/{month}/all_gee_added/{analysis_year}/train.data.nc'
y_train = f'{root_dir}/{month}/all_gee_added/{analysis_year}/train.target.nc'

# --- Dealing with testing data ---
testing_x = f'{root_dir}/{month}/all_gee_added/{analysis_year}/test.data.nc'
testing_y = f'{root_dir}/{month}/all_gee_added/{analysis_year}/test.target.nc'

# Read in toar testing target
toar_test = xr.open_dataset(testing_y)
toar_test = format_lon(toar_test)
y = toar_test.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))
y_test = y['target'].values
# Remove Nans
mask_testing = ~np.isnan(y_test)
y_test = y_test[mask_testing]

data_test = xr.open_dataset(testing_x)
data_test = format_lon(data_test)

if add_gee:
    modis_train = f'{root_dir}/{month}/all_gee_added/{analysis_year}/train.modis.nc'
    #fire_train = f'{root_dir}/{month}/{analysis_year}/train.fire.nc'
    fire_train = None
    pop_train = f'{root_dir}/{month}/all_gee_added/{analysis_year}/train.pop.nc'

    # Deal with gee testing data
    modis_test = f'{root_dir}/{month}/all_gee_added/{analysis_year}/test.modis.nc'
    #fire_test = f'{root_dir}/{month}/{analysis_year}/test.fire.nc'
    pop_test = f'{root_dir}/{month}/all_gee_added/{analysis_year}/test.pop.nc'
    modis_ds_test = xr.open_dataset(modis_test)
    #fire_ds_test = xr.open_dataset(fire_test)
    pop_ds_test = xr.open_dataset(pop_test)
    #combined_test = data_test.merge(modis_ds_test).merge(fire_ds_test).merge(pop_ds_test)
    combined_test = data_test.merge(modis_ds_test).merge(pop_ds_test)
else:
    modis_train, fire_train, pop_train = None, None, None
    add_gee = False
    combined_test = data_test

# Deal with training data
X, y = generate_X_and_y(x_train, y_train, modis_train, fire_train, pop_train, add_gee)

var_names = list(combined_test.keys())
combined_test = combined_test.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))
X_test = []
for j in tqdm(range(len(var_names))):
    x = combined_test[var_names[j]].values[mask_testing]
    X_test.append(x)
X_test = np.column_stack(X_test)

if param_tuning:
    Logger.info('Finding optimal hyperparameters')
    params = [{
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 4, 5]
            }]
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
    clf = GridSearchCV(
        RandomForestRegressor(), params, cv=kfold,
        scoring='neg_root_mean_squared_error', n_jobs=10, error_score='raise'
    )
    clf.fit(X, y)
    print('Parameter selection:')
    print(f'n_estimators: {clf.best_params_["n_estimators"]}')
    print(f'max_depth: {clf.best_params_["max_depth"]}')

    # Create the Gradient Boosting predictor
    Logger.info('Training Random Forest')
    rf = RandomForestRegressor(n_estimators=clf.best_params_['n_estimators'],
                               max_depth=clf.best_params_['max_depth'],
                               random_state=300,
                               verbose=1)
else:
    rf = RandomForestRegressor(n_estimators=100,
                               max_features=int(0.3*(len(var_names))),
                               random_state=300,
                               verbose=1)

rf.fit(X, y)
out_model = os.path.join(results_dir, 'random_forest_predictor.joblib')
dump(rf, out_model)

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

with open('{}/results.txt'.format(results_dir), 'w') as f:
    f.write(' rmse is {}'.format(rmse))
    f.write(' mean absolute percentage error is: {}'.format(mape))
    f.write(" r correlation is: {}".format(r))
    f.write(' r2 score is: {}'.format(r2))

# Calculate importances
importances = calc_importances(rf, var_names, results_dir)

# Calculate permutation importance
perm_importances = calc_perm_importance(rf, X_test, y_test, var_names, results_dir)

# Plot importances
plot_importances(importances, perm_importances, results_dir, month, analysis_year, aoi)

# Plot true vs predicted
Logger.info('Plotting y_test vs y_hat')
truth_vs_predicted(y_test, yhat, results_dir, aoi)

# Plot histogram
plot_histogram(y_test, yhat, results_dir, aoi)