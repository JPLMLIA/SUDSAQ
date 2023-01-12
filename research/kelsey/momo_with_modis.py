'''
OLD: Moved over to modis_processing --> will delete this eventually
Running RF experiment using momo + modis data
@author: kdoerksen
'''

import os, glob
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xarray as xr
from tqdm import tqdm
import pandas as pd
import ee
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# example lat, lon: -87.732457, 41.670992

def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)

def process_collection(collection, lc, lat, lon):
    '''
    Process GEE collection to return median of date range
    '''
    lc_mean = []
    for k in range(len(lat)):
        print('Processing for lat {} for all lon'.format(lat[k]))
        for m in range(len(lon)):
            point = ee.Geometry.Point(lon[m], lat[k])
            roi = point.buffer(50000)  # 50km buffer will match the 100x100km momo grid

            img_area = collection.filterBounds(roi)
            img_med = img_area.median()

            band_arrs = img_med.sampleRectangle(region=roi)
            band_arr = band_arrs.get(lc)
            try:
                np_arr = np.array(band_arr.getInfo())
                lc_mean.append(np_arr.mean())
            except:
                lc_mean.append(np.nan)

    return lc_mean

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

region = 'west_na'
bbox = bbox_dict[region]

root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'
if not os.path.exists(root_dir):
    root_dir = '/Users/kelseydoerksen/code/suds-air-quality/kelsey_data'

script_home = '/Users/kelseydoerksen/code/suds-air-quality/research/kelsey/run_rf.py'

# set the directory for that month
models_dir = f'{root_dir}/models/2011-2015/bias-8hour/'
training_x = []
training_y = []

# Set month of interest
month = 'jul'
# Set test year, train years
test_year = [2011]
train_years = [2012, 2013, 2014, 2015]

for year in train_years:
    training_x.append('{}/{}/{}/train.data.nc'.format(root_dir, month, year))
    training_y.append('{}/{}/{}/train.target.nc'.format(root_dir, month, year))

testing_x = ['{}/{}/{}/test.data.nc'.format(root_dir, month, test_year[0])]
testing_y = ['{}/{}/{}/test.target.nc'.format(root_dir, month, test_year[0])]

ds = xr.open_dataset(testing_y[0])
ds = format_lon(ds)
ds = ds.sel(lat=slice(bbox[2], bbox[3]), lon=slice(bbox[0], bbox[1]))
lat = ds['lat'].values
lon = ds['lon'].values

# Authenticate ee
ee.Authenticate()

# Initialize ee
ee.Initialize()

# Specify land cover type
lc = 'LC_Type1'

# Dates of analysis
dates = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01']


for i in range(len(dates)-1):
    # Grab modis data
    dataset = ee.ImageCollection('MODIS/006/MCD12Q1').select(lc).filterDate(dates[i], dates[i+1])
    import ipdb
    ipdb.set_trace()
    test = process_collection(dataset, lc, lat, lon)
    import ipdb
    ipdb.set_trace()


'''

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