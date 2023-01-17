#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kdoerksen
K-means clustering analysis on momo-chem features
"""
import urllib
import numpy as np
import os, glob
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)

def calc_cluster_means(X, y, k, cluster_model):
    '''
    Calculate the mean of each cluster
    '''
    df = pd.DataFrame()
    bias_means = []
    feature_means = []
    cluster_num = []
    for i in range(k):
        result = np.where(cluster_model.labels_ == i)
        bias_means.append(np.mean(y[result]))
        feature_means.append(np.mean(X[:,0][result]))
        cluster_num.append(i)

    df['feature_mean'] = feature_means
    df['bias_mean'] = bias_means
    df['cluster'] = cluster_num

    return df


root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'
if not os.path.exists(root_dir):
    root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'

script_home = '/Users/kelseydoerksen/code/suds-air-quality/research/kelsey/run_rf.py'

month = 'jul'
years = [2011, 2012, 2013, 2014, 2015]

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
# Grab from testing set, smaller to work with
testing_x = np.hstack(glob.glob(f'{root_dir}/{month}/*/test.data.nc'))

# If [#] at the end, grabbing only one file to work with
testing_y = glob.glob(f'{root_dir}/{month}/*/test.target.nc')[0]

# Read in toar testing target (predictand_
toar_test = xr.open_dataset(testing_y)
y_test = toar_test['target'].values
# Remove Nans
mask_testing = ~np.isnan(y_test)
y_test = y_test[mask_testing]

# If [#] at the end, grabbing only one file to work with
data = xr.open_dataset(testing_x[0])
'''
# top momo features from previous analysis
momo_features = ['momo.2dsfc.C3H7OOH', 'momo.osrc',
                 'momo.ch2o', 'momo.ps', 'momo.pan',
                 'momo.2dsfc.CH2O',
                 'momo.hno3', 'momo.co', 'momo.aerosol.nh4',
                 'momo.oh', 'momo.ssrc', 'momo.2dsfc.NH3',
                 'momo.aerosol.no3', 'momo.sens', 'momo.t',
                 'momo.slrc', 'momo.2dsfc.N2O5',
                 'momo.so2', 'momo.2dsfc.DCDT.SO2']


for feature in momo_features:
    print('Plotting feature: {}'.format(feature))
    X = []
    x = data['{}'.format(feature)].values[mask_testing]
    X.append(x)
    X = np.column_stack(X)

    plt.plot(X,y_test,'.')
    plt.xlabel('{}'.format(feature))
    plt.ylabel('Bias (target)')
    plt.title('Scatter plot of {} and bias'.format(feature))
    plt.show()
'''

potential_features = ['momo.ssrc', 'momo.sens', 'momo.t', 'momo.slrc']

# Build clustering model
clustering = KMeans(n_clusters=3,random_state=5)
colors = np.array(["Red","Green","Blue"])

for feature in potential_features:
    X = []
    x = data['{}'.format(feature)].values[mask_testing]
    X.append(x)
    X = np.column_stack(X)
    clustering.fit(X)
    print('Cluster Means for feature {}'.format(feature))
    print(calc_cluster_means(X, y_test, 3, clustering))
    plt.scatter(x=X ,y=y_test, c = colors[clustering.labels_],s=50)
    plt.title("K means Clustering for feature {}".format(feature))
    plt.show()

'''
# ------- 
# Let's add regions as cluster labels to see what the data looks like
bbox_dict = {'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80]}

# Read in toar testing target
toar_test = xr.open_dataset(testing_y)
toar_test = format_lon(toar_test)

df_list = []
count = 1
for aoi in bbox_dict.keys():
    bbox = bbox_dict[aoi]

    y = toar_test.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))
    y_test = y['target'].values
    # Remove Nans
    mask_testing = ~np.isnan(y_test)
    y_test = y_test[mask_testing]
    # Create dataframe
    df = pd.DataFrame()
    df['y'] = y_test
    df['region'] = count
    count+=1
    df_list.append(df)


df_all = pd.concat([df_list[0], df_list[1], df_list[2], df_list[3]])
y = df_all['y'].to_numpy()
X = df_all['region'].to_numpy()

X = np.column_stack(X)
y = np.column_stack(y)
X = np.moveaxis(X, [0, 1], [1, 0])
y = np.moveaxis(y, [0, 1], [1, 0])


clustering = KMeans(n_clusters=4,random_state=37)
clustering.fit(X)
plt.scatter(x=X ,y=y,s=50)
plt.show()
'''