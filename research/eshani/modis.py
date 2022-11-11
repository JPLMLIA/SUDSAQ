import ee
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from IPython.display import Image
import requests

import matplotlib as mpl
from matplotlib import cm
#from colorspacious import cspace_converter
from matplotlib.colors import ListedColormap
from sctriangulate.colors import build_custom_continuous_cmap

# color map for land cover type 1
cmap1 = build_custom_continuous_cmap([5, 69, 10], [8, 106, 16], [84, 167, 8], [120, 210, 3],
                                        [0, 153, 0], [198, 176, 68], [220, 209, 89], [218, 222, 72], 
                                        [251, 255, 19], [182, 255, 5], [39, 255, 135], [194, 79, 68],
                                        [165, 165, 165], [255, 109, 76], [105, 255, 248], [249, 255, 164], [28, 13, 255])

# color map for land cover type 2
cmap2 = build_custom_continuous_cmap([28, 13, 255], [5, 69, 1], [8, 106, 16], [84, 167, 8], 
                                      [120, 210, 3], [0, 153, 0], [198, 176, 68], [220, 209, 89],
                                      [218, 222, 72], [251, 255, 19], [182, 255, 5], [39, 255, 135],
                                      [194, 79, 68], [165, 165, 165], [255, 109, 76], [249, 255, 164])

# color map for land cover type 3
cmap3 = build_custom_continuous_cmap([28, 13, 255], [182, 255, 5], [220, 209, 89], [194, 79, 68],
                                      [251, 255, 19], [8, 106, 16], [120, 210, 3], [5, 69, 10], 
                                      [84, 167, 8], [249, 255, 164], [165, 165, 165])

# color map for land cover type 4
cmap4 = build_custom_continuous_cmap([28, 13, 255], [5, 69, 10], [8, 106, 16], [84, 167, 8],
                                      [120, 210, 3], [0, 153, 0], [182, 255, 5], [249, 255, 164], [165, 165, 165])

# color map for land cover type 5
cmap5 = build_custom_continuous_cmap([28, 13, 255], [5, 69, 10], [8, 106, 16], [84, 167, 8],
                                      [120, 210, 3], [220, 209, 89], [182, 255, 5], [218, 222, 72], 
                                      [194, 79, 68], [165, 165, 165], [105, 255, 248], [249, 255, 164])

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

lc = 'LC_Type1'

# Initial date of interest (inclusive).
i_date = '2005-01-01'

# Final date of interest (exclusive).
f_date = '2006-01-01'

# Define an image.
dataset = ee.ImageCollection('MODIS/006/MCD12Q1').select(lc).filterDate(i_date, f_date)

# Reduce the LST collection by mean.
img = (dataset.median()
  .multiply(0.0001)
  .setDefaultProjection(dataset.first().projection()))

# Define an area of interest.
aoi = ee.Geometry.Polygon(
  [[[-110.8, 44.7],
    [-110.8, 44.6],
    [-110.6, 44.6],
    [-110.6, 44.7]]], None, False)

# Define the urban location of interest as a point near `Lyon`, France.
u_lon = 4.8148
u_lat = 45.7758
u_poi = ee.Geometry.Point(u_lon, u_lat)

# Define a region of interest with a buffer zone of 1000 km around Lyon.
roi = u_poi.buffer(100000)  #need to 50-60 km 
#1 degree is 111 km

# Get 2-d pixel array for AOI - returns feature with 2-D pixel array as property per band.
band_arrs = img.sampleRectangle(region=roi)

# Get individual band arrays.
band_arr = band_arrs.get(lc)

# Transfer the arrays from server to client and cast as np array.
np_arr = np.array(band_arr.getInfo())

#plot_color_gradients()
plot = plt.imshow(np_arr, cmap=cmap1) 
plt.savefig(f'{lc}_plot.png')
''' 
img = (dataset.median().multiply(0.0001).setDefaultProjection(dataset.first().projection()))
lc_mean = []
lc_std = []

#long/lat specified by aq_bench

for i in range(len(longitude/latitude array)):
  point = ee.Geometry.Point(lon[i], lat[i])
  roi = u_poi.buffer(50000)  #need to 50-60 km
  band_arrs = img.sampleRectangle(region=roi)
  band_arr = band_arrs.get(lc)
  np_arr = np.array(band_arr.getInfo())
  lc_mean.append(np_arr.mean())
  lc_std.append(np_arr.std())

x = np.array(np.column_stack([x, lc_mean, lc_std]))
'''
