import math
import os
from turtle import xcor
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

VARS = ['o3_average_values', 'o3_daytime_avg', 'o3_nighttime_avg', 'o3_median',
        'o3_perc25', 'o3_perc75', 'o3_perc90', 'o3_perc98', 'o3_dma8eu',
        'o3_avgdma8epax', 'o3_drmdmax1h', 'o3_w90', 'o3_aot40', 'o3_nvgt070',
        'o3_nvgt100']
        
NUM_FEATURES = ['alt', 'relative_alt', 'water_25km',
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
OTHER = ['id', 'country', 'htap_region', 'dataset', 'lon', 'lat']

# reads in csv file
data = pd.read_csv('AQbench_dataset.csv')

# filters data to only include North American data stations
lon = np.array(data['lon'])
lat = np.array(data['lat'])
mask_lon = (lon>=-180) & (lon<=-53) #masks
mask_lat = (lat>=-7) & (lat<=83)
mask = mask_lon & mask_lat
data = data.iloc[mask]


feature_names = np.array(NUM_FEATURES)

# one-hot encodes the categorical vairables
x_cat = np.zeros((data.shape[0], len(CAT_FEATURES)))
x_range = range(len(CAT_FEATURES))
for i in x_range:
    x = np.array(data[CAT_FEATURES[i]])
    unique_cat = np.unique(x)
    feature_names = np.concatenate([feature_names, unique_cat])
    for j, u in enumerate(unique_cat):
         mask = x== u
         x_cat[mask, i] = j+1
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(data[CAT_FEATURES])

# sets y to be 5 year average ozone
y = np.array(data['o3_average_values']) 
# sets x to be numerical features and one-hot-encoded categorical features
x = np.array(np.column_stack([np.array(data[NUM_FEATURES]), onehot]))

# Random Forest Model ran on 5 splits of data to generate pred_y
# kfold = GroupKFold(n_splits=len(data_files))
pred_y = np.zeros((len(y), ))
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
for train_index, test_index in kfold.split(x):
    train_x, test_x = x[train_index], x[test_index]
    train_y, test_y = y[train_index], y[test_index]
    train_mask, test_mask = mask[train_index], mask[test_index]

    rf_predictor = RandomForestRegressor(max_depth=None, min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=51, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=True, n_jobs=None, random_state=12, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
    rf_predictor.fit(train_x[train_mask],
                     train_y[train_mask])

    pred_y[test_index] = rf_predictor.predict(test_x)

# calculates root mean squared error
mse = sklearn.metrics.mean_squared_error(y, pred_y)
rmse = math.sqrt(mse)
print("Root Mean Sqaured Error = " + str(rmse))

# calculates r correlation value
r = np.corrcoef(y, pred_y)
print("r correlation = " + str(r[0,1]))

# creates scatter plot of pred_y vs y and plots line of best fit
a, b = np.polyfit(y, pred_y, 1)
plt.scatter(y, pred_y, s=3)
plt.plot(y, a*y+b, color="red", linewidth=2)
plt.title('Predicted Ozone vs Actual Ozone in North America', fontsize=12)
plt.xlabel('Actual Ozone (ppb)', fontsize=12)
plt.ylabel('Predicted Ozone (ppb)', fontsize=12)
plt.xlim([0, 60])
plt.ylim([0, 60])
plt.savefig('scatter_north_america.png')
plt.show()

# plots histogram of y and pred_y
range1 = np.max(y)-np.min(y)
range2 = np.max(pred_y)-np.min(pred_y)
plt.hist(y, bins=(int)(range1/2), color="blue", alpha=0.5)
plt.hist(pred_y, bins=(int)(range2/2), color="red", alpha=0.5)
plt.title('Histogram of Actual and Predicted Average Ozone in North America', fontsize=12)
plt.xlabel('Average Ozone over 5 years ', fontsize=12)
legend_drawn_flag = True
plt.legend(["actual", "predicted"], loc=0, frameon=legend_drawn_flag)
plt.savefig('hist_north_america.png')
plt.show()


# Calculates feature importance
X, y = make_classification(n_samples=len(y), n_features=x.shape[1], n_informative=5, n_redundant=5, random_state=1)
model = RandomForestClassifier()
model.fit(x, y)
importance = model.feature_importances_
# normalizes feature importance
normal_array = importance/np.max(importance)
plt.bar(feature_names, normal_array, color='blue', alpha=0.5)


# Calculates permutation importance
r = permutation_importance(model, x, y,
                            n_repeats=30,
                            random_state=0)
# normalizes permutation importance
normal_array = r.importances_mean/np.max(r.importances_mean)
plt.bar(feature_names, normal_array, color='red', alpha=0.5)

# plots normalized feature importance against normalized permuation importance
plt.title('Feature Importance vs Permutation Importance of North America', fontsize=12)
plt.xlabel('Feature Name', fontsize=12)
plt.ylabel('Normalized Importance', fontsize=12)
plt.xticks(fontsize=8, rotation = 90)
legend_drawn_flag = True
plt.legend(["Feature Importance", "Permutation Importance"], loc=0, frameon=legend_drawn_flag)
plt.savefig('importance_north_america.png')
plt.show()
