import ee
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sctriangulate.colors import build_custom_continuous_cmap
import math
import os
from turtle import xcor
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from tqdm import tqdm

# color map for land cover type 1
cmap1 = build_custom_continuous_cmap([5, 69, 10], [8, 106, 16], [84, 167, 8], [120, 210, 3],
                                        [0, 153, 0], [198, 176, 68], [220, 209, 89], [218, 222, 72], 
                                        [251, 255, 19], [182, 255, 5], [39, 255, 135], [194, 79, 68],
                                        [165, 165, 165], [255, 109, 76], [105, 255, 248], [249, 255, 164], [28, 13, 255])

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

lc = 'LC_Type1'

# Initial date of interest (inclusive).
i_date = '2010-01-01'

# Final date of interest (exclusive).
f_date = '2015-01-01'

# Define an image.
dataset = ee.ImageCollection('MODIS/006/MCD12Q1').select(lc).filterDate(i_date, f_date)

# Reduce the LST collection by median.
img = (dataset.median().multiply(0.0001).setDefaultProjection(dataset.first().projection()))

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

feature_names = np.hstack([np.array(NUM_FEATURES), 'lc_mean', 'lc_std', 'lc_mode', 'lc_counts'])

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

#defines arrays for 4 new featurs 
lc_mean = []
lc_std = []
lc_mode = []
lc_counts = []

# gets AQ-bench lon and lat stations
lon = np.array(data['lon'])
lat = np.array(data['lat'])

# checks to see if data file is generated otherwise generates data file
if(os.path.isfile('ee_image_data.csv')):
    extra_features = np.loadtxt('ee_image_data.csv')
else:
    # iterates through each point and get 50 km image around it in numpy array format
    # finds mean, std, mode, and num of unique values to add to arrays
    for i in tqdm(range(len(lon))):
        point = ee.Geometry.Point(lon[i], lat[i]) 
        roi = point.buffer(50000)  #need to 50 km
        band_arrs = img.sampleRectangle(region=roi)
        band_arr = band_arrs.get(lc)
        #try and except for boundry points where arrays can't be generated so nan is added instead
        try: 
            np_arr = np.array(band_arr.getInfo()) 
            lc_mean.append(np_arr.mean())
            lc_std.append(np_arr.std())
            values, counts = np.unique(np_arr, return_counts=True)
            m = counts.argmax()
            lc_mode.append(values[m])
            lc_counts.append(len(values))
        except:
            print(i)
            print(point)
            lc_mean.append(np.nan)
            lc_std.append(np.nan)
            lc_mode.append(np.nan)
            lc_counts.append(np.nan)
    # stores extracted features into a csv file       
    lc_mean = np.hstack(lc_mean)
    lc_std = np.hstack(lc_std)
    lc_mode = np.hstack(lc_mode)
    lc_counts = np.hstack(lc_counts)
    extra_features = np.column_stack([lc_mean, lc_std, lc_mode, lc_counts])
    np.savetxt('ee_image_data.csv', extra_features)

# add modis features to x
x = np.array(np.column_stack([x, extra_features]))

# applies mask to filter out any stations with nan values
mask_nan = ~np.isnan(np.sum(x, axis=1))
print(mask_nan.shape)
x = x[mask_nan]
y = y[mask_nan]

# Random Forest Model ran on 5 splits of data to generate pred_y
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
print("Root Mean Squared Error = " + str(rmse))

# calculates r correlation value
r = np.corrcoef(y, pred_y)
print("r correlation = " + str(r[0,1]))


# creates scatter plot of pred_y vs y and plots line of best fit
a, b = np.polyfit(y, pred_y, 1)
plt.scatter(y, pred_y, s=3)
plt.plot(y, a*y+b, color="red", linewidth=2)
plt.title('Predicted Ozone vs Actual Ozone (Earth Engine)', fontsize=12)
plt.xlabel('Actual Ozone (ppb)', fontsize=12)
plt.ylabel('Predicted Ozone (ppb)', fontsize=12)
plt.xlim([0, 60])
plt.ylim([0, 60])
plt.savefig('scatter_ee.png')
plt.show()


# plots histogram of y and pred_y
range1 = np.max(y)-np.min(y)
range2 = np.max(pred_y)-np.min(pred_y)
plt.hist(y, bins=(int)(range1/2), color="blue", alpha=0.5)
plt.hist(pred_y, bins=(int)(range2/2), color="red", alpha=0.5)
plt.title('Histogram of Actual and Predicted Average Ozone over 5 years (Earth Engine)', fontsize=12)
plt.xlabel('Average Ozone over 5 years', fontsize=12)
legend_drawn_flag = True
plt.legend(["actual", "predicted"], loc=0, frameon=legend_drawn_flag)
plt.savefig('hist_ee.png')
plt.show()


# plots y-pred_y on a map using TOAR longitudinal and latiduninal station points
ig, ax = plt.subplots(figsize=(18, 9),
                        subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

sc = ax.scatter(np.array(data['lon']), np.array(data['lat']), c=y-pred_y, cmap="jet", s=3)
plt.title('5-year ozone average y-pred_y (Earth Engine)', fontsize=12)
plt.colorbar(sc)
plt.savefig('OzoneMap_ee_residual.png')
plt.show()

# Calculates feature importance
X, y = make_classification(n_samples=len(y), n_features=x.shape[1], n_informative=5, n_redundant=5, random_state=1)
model = RandomForestClassifier()
model.fit(x, y)
importance = model.feature_importances_
normal_array = r.importances_mean/np.max(r.importances_mean)

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
plt.title('Feature Importance vs Permutation Importance (Earth Engine)', fontsize=12)
plt.xlabel('Feature Name', fontsize=12)
plt.ylabel('Normalized Importance', fontsize=12)
plt.xticks(fontsize=8, rotation = 90)
legend_drawn_flag = True
plt.legend(["Feature Importance", "Permutation Importance"], loc=0, frameon=legend_drawn_flag)
plt.savefig('importance_ee.png')
plt.show()
