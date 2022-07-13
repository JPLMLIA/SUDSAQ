#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

# pip install xarray dask
# Dask is needed for lazy loading via xr.open_mfdataset

#%%
import xarray as xr

from glob import glob
#%%
path  = '/Volumes/MLIA_active_data/data_SUDSAQ/data'
momo  = glob(f'{path}/momo/**/*.nc')                    # List of MOMO .nc files
toar  = glob(f'{path}/toar/matched/**/*.nc')            # List of TOAR .nc files
files = momo + toar
len(files)
#%%
ds = xr.open_mfdataset(momo, engine='scipy', parallel=True) # Lazy load in parallel
ds

#%%

ns = ds.sel(time='2012-06') # Select only June of 2012
ns = ns[['co', 'toar/o3/dma8epa/mean', 'toar/o3/dma8epa/std', 'toar/o3/dma8epa/count']] # Select only these variables
ns

#%%

ns.load() # Now load the data into memory
ns

#%%
# The times of TOAR do not align with the times of MOMO
(~ns.dropna('time', subset=['toar/o3/dma8epa/count']).isnull()).any()

#%%
# Resample
ts = ns.groupby('time.day').mean()
ts

#%%

stacked = ts.stack({'stacked': ['lat', 'lon']})
stacked

#%%

da = ts.to_array()
da.values

#%%

da.stack({'data': ['day', 'variable']})
da.stack({'data': ['day', 'variable'], 'loc': ['lat', 'lon']})

#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def geospatial(data, title=None):
    """
    """
    # Prepare figure
    fig = plt.figure(figsize=(15, 8))
    ax  = plt.subplot(projection=ccrs.PlateCarree())

    # data.plot.pcolormesh(x='lon', y='lat', ax=ax, levels=10, cmap='viridis')
    data.plot(ax=ax)

    ax.coastlines()
    ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)

    if title:
        ax.set_title(title)
    plt.show()

#%%
# Simple plotting process
ax = plt.subplot(projection=ccrs.PlateCarree())
ts['co'].mean('day').plot(ax=ax) # Data must be 2d, so take the mean of the `days` dimension
ax.coastlines()
ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)

#%%

geospatial(ts['co'].mean('day'))
geospatial(ts['toar/o3/dma8epa/mean'].mean('day'))
geospatial(ts['toar/o3/dma8epa/std'].mean('day'))
geospatial(ts['toar/o3/dma8epa/count'].mean('day'))

#%%
# Convert to numpy
ds.to_array().values

type(ts['toar/o3/dma8epa/mean'].values)

#%% [Markdown]
## # Subselect MOMO data between 8am and 4pm and take the average to align with daily TOAR data

#%%
import xarray
import datetime as dt

# Load in the data
ds = xr.open_mfdataset(files, engine='scipy', parallel=True)
ns = ds.sel(time='2012-06')
ns = ns[['co', 'toar/o3/dma8epa/mean', 'toar/o3/dma8epa/std', 'toar/o3/dma8epa/count']]
ns.load()

#%%

time = ns.time.dt.time # Convert the times dimension to datetime.time objects
mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0))
# Select timestamps between 8am and 4pm OR midnight

ss = ns.where(mask, drop=True)     # Drop timestamps that don't match
gs = ss.groupby('time.day').mean() # Take the mean by day
gs

#%%

# Doing the same as above but without selecting midnight

# Only select between 8 am and 4 pm
mask = (dt.time(8) < time) & (time < dt.time(16))
ss   = ns.where(mask, drop=True)

# TOAR is entirely NaN because its values are only at midnight which wasn't selected
(ss['toar/o3/dma8epa/mean'].isnull()).all().values

# `ss` is the 8am-4pm subselect, take the daily mean of `co`
co   = ss['co'].groupby('time.day').mean()
# Take the daily mean of the toar variables
toar = ns.drop('co').groupby('time.day').mean()

ns.drop('co')

# Merge them together (automatically on the `day` dimension created by the groupby('time.day'))
gs = xr.merge([co, toar])

#%%
#%%
#%%
import datetime as dt

ns = ds.sel(time=slice('2010-07', '2014-07'))
ns
ns = ns[['so2', 'toar/o3/dma8epa/count']]

time = ns.time.dt.time # Convert the times dimension to datetime.time objects
mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0))
# Select timestamps between 8am and 4pm OR midnight

ss = ns.where(mask, drop=True)     # Drop timestamps that don't match
ss.load()
#%%
rs = ss.resample(time='1D').mean()
rs

#%%

ns = ds.sel(time=(ds.time.dt.month == 6))
ns = ns.sel(time=slice('2010', '2014'))
ns = ns[['so2', 'toar/o3/dma8epa/count']]

time = ns.time.dt.time
mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0))

ss = ns.where(mask, drop=True)
ss.load()

rs = ss.resample(time='1D').mean()
rs = rs.sel(time=(rs.time.dt.month == 6))
rs
np.unique(rs.time.dt.year)

fs = rs.dropna('time')

help(rs.resample)

#%%

ss.resample(time='1D', skipna=True).mean()

#%%


#%%

for year, yds in fs.groupby('time.year'):
    break
year
yds

#%%

l = list(fs.groupby('time.year'))
l[0]

fs = rs.dropna('time')

years = dict(fs.groupby('time.year'))

years.keys()
years[2010]

xr.concat([years[2010], years[2010]], dim='time')

#%%
ls /Volumes/MLIA_active_data/data_SUDSAQ/MOMO/outputs
ls /Volumes/MLIA_active_data/data_SUDSAQ/MOMO/mda8
#%%

ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/MOMO/mda8/*.nc', engine='scipy', parallel=True)
# ds = ds.rename({'mda8': 'mda8/daily'})

ns = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/MOMO/outputs/2hr_o3_2*.nc', engine='scipy', parallel=True)

ms = xr.merge([ds, ns])
ms

ss = ms.sel(time='2005-01')
ss
ss.load()
#%%
with xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/01.nc', mode='a') as os:
    fs = xr.merge([os, ss])

#%%
fs.to_netcdf('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/01.nc', mode='a')
#%%

ss.to_netcdf('/data/MLIA_active_data/data_SUDSAQ/data/momo/2005/2.01.nc', mode='a', engine='scipy')

1

#%%

def load():
    ds = xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/MOMO/mda8/*.nc', engine='scipy', parallel=True)
    ns = xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/MOMO/outputs/2hr_o3_2*.nc', engine='scipy', parallel=True)
    ms = xr.merge([ds, ns])
    return ms

    ss = ms.sel(time='2005-01')
    ss

#%%
import xarray as xr

from tqdm import tqdm

def load_ozone():
    ds = xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/MOMO/mda8/*.nc', engine='scipy', parallel=True)
    ns = xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/MOMO/outputs/2hr_o3_2*.nc', engine='scipy', parallel=True)
    ms = xr.merge([ds, ns])
    return ms

ozone = load_ozone()

for year, yds in tqdm(ozone.groupby('time.year'), desc='Processing Years', position=1):
    for month, mds in tqdm(yds.groupby('time.month'), desc='Processing Months', position=0):
        mds.to_netcdf(f'/data/MLIA_active_data/data_SUDSAQ/data/momo/{year}/{month:02}.nc', mode='a', engine='scipy')

#%%

ds = xr.open_dataset(f'/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2012/02.nc', mode='r', engine='scipy')
ns = ds[['mda8', 'o3', 't']]


#%%

ns.load()

#%%

ns['mda8'].dropna('time').time
ns['t'].dropna('time').time

#%%
import datetime as dt
import numpy    as np
import xarray   as xr

ds = xr.open_mfdataset([
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2012/02.nc',
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/02.nc'
    ], mode='r', engine='scipy')

ns = ds[['mda8', 'o3', 't', 'toar/o3/dma8epa/count']]
ns.load()
#%%
time = ns.time.dt.time
mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0)) | (time == dt.time(1))
#                  8am < time < 4pm               or time == 12am        or time == 1am

ss = ns.where(mask, drop=True)

# Remove 1 am for MOMO variables by setting to NaN
ss[['o3', 't']].loc[{
    'time': ss.time.dt.time == dt.time(1)
}] = np.nan

rs = ss.resample(time='1D').mean().dropna('time')
rs

#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def geospatial(data, title=None):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    # ax.set_extent([-20, 20, 30, 60], ccrs.PlateCarree())
    data.plot.pcolormesh(x='lon', y='lat', ax=ax, levels=10, cmap='viridis', vmin=3, vmax=40)
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)
    if title:
        ax.set_title(title)
    plt.show()

geospatial(ns['toar/o3/dma8epa/count'].mean('time'))

#%%

import dask

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    ds = xr.open_mfdataset(f'/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/**/*.nc', engine='scipy', parallel=True)

#%%

ds = xr.open_mfdataset(f'/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2017/*.nc', engine='scipy', parallel=True)
ds

#%%

ds = xr.open_mfdataset([
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/v2/2012/06.nc'
    ], mode='r', engine='scipy')

ns = ds[['toar/o3/dma8epa/count']]
ns.load()

#%%

files = []
for i in range(5, 18):
    files += glob(f'/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/20{i:02}/*.nc')

ds = xr.open_mfdataset(files, engine='scipy', parallel=True)
ds
