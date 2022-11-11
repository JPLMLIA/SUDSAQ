import ee
import pandas as pd
import numpy as np

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

lc = 'LC_Type1'

# Dates of analysis
dates = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01']

# Define an image.
dataset = ee.ImageCollection('MODIS/006/MCD12Q1').select(lc).filterDate(dates[0], dates[1])

# Reduce the LST collection by median.
img = dataset.first()

# reads in csv file
data = pd.read_csv('/Volumes/MLIA_active_data/data_SUDSAQ/AQ-Bench/AQbench_dataset.csv')

# defines arrays for 4 new features
lc_mean = []
lc_std = []
lc_mode = []
lc_counts = []

# gets AQ-bench lon and lat stations
lon = np.array(data['lon'])
lat = np.array(data['lat'])


for i in range(len(lon)):
    point = ee.Geometry.Point(lon[i], lat[i])
    roi = point.buffer(50000)  # need to 50 km
    band_arrs = img.sampleRectangle(region=roi)
    band_arr = band_arrs.get(lc)
    import ipdb
    ipdb.set_trace()
    # try and except for boundry points where arrays can't be generated so nan is added instead
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
#np.savetxt('ee_image_data.csv', extra_features)