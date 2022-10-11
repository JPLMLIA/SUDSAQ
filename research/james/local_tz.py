#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import xarray as xr

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2009/01.nc',
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2010/01.nc'
    # '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc',
    # '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/metadata.nc'
]

ds = xr.open_mfdataset(files, parallel=True)
ds = ds[['momo.t', 'momo.u', 'momo.v']]

# ds.load()
#%%
from sudsaq.data import Dataset

ds = Dataset(ds)

ds

#%%
from tqdm import tqdm
import datetime as dt
import numpy as np

timezones = [
#  offset, (west, east)
    (  0, (  0.0, 7.5)),
    (  1, (  7.5, 22.5)),
    (  2, ( 22.5, 37.5)),
    (  3, ( 37.5, 52.5)),
    (  4, ( 52.5, 67.5)),
    (  5, ( 67.5, 82.5)),
    (  6, ( 82.5, 97.5)),
    (  7, ( 97.5, 112.5)),
    (  8, (112.5, 127.5)),
    (  9, (127.5, 142.5)),
    ( 10, (142.5, 157.5)),
    ( 11, (157.5, 172.5)),
    ( 12, (172.5, 180.0)),
    (-12, (180.0, 187.5)),
    (-11, (187.5, 202.5)),
    (-10, (202.5, 217.5)),
    ( -9, (217.5, 232.5)),
    ( -8, (232.5, 247.5)),
    ( -7, (247.5, 262.5)),
    ( -6, (262.5, 277.5)),
    ( -5, (277.5, 292.5)),
    ( -4, (292.5, 307.5)),
    ( -3, (307.5, 322.5)),
    ( -2, (322.5, 337.5)),
    ( -1, (337.5, 352.5)),
    (  0, (352.5, 360.0))
]

data = []
for offset, (west, east) in tqdm(timezones):
    sub  = ds.sel(lon=slice(west, east))
    time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time

    mask = ( dt.time(8) < time) & (time < dt.time(16) )

    sub = sub.where(mask, drop=True)
    sub.coords['time'] = sub.time.dt.floor('1D')
    sub = sub.groupby('time').mean()

    data.append(sub)

ns = xr.merge(data)
ns

# 27 seconds

#%%

ns.load() # 5.5 seconds

#%%

for offset, (west, east) in tqdm(timezones):
    sub = ds.sel(lon=slice(west, east))
    sub.coords['time'] = sub.time + np.timedelta64(offset, 'h')
    break

mask
sub

#%%

for offset, (west, east) in tqdm(timezones):
    sub  = ds.sel(lon=slice(west, east))
    time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time
    mask = ( dt.time(8) <= time) & (time < dt.time(16) )
    # break
    if offset==1:
        break

time
mask

#%%

# 76,390,400
#  7,936,000
76390400/360/160/5/5

mask

320*160*(5*365)

31*5*160*320

#%%

def select_times(ds, sel, time):
    """
    Selects timestamps using integer hours (0-23) over all dates
    """
    if isinstance(sel, list):
        mask = (dt.time(sel[0]) <= time) & (time < dt.time(sel[1]))
        ds   = ds.where(mask, drop=True)

        # Floor the times to the day for the groupby operation
        ds.coords['time'] = ds.time.dt.floor('1D')

        # Now group as daily taking the mean
        ds = ds.groupby('time').mean()
    else:
        mask = (time == dt.time(sel))
        ds   = ds.where(mask, drop=True)

        # Floor the times to the day for the groupby operation
        ds.coords['time'] = ds.time.dt.floor('1D')

    return ds

time = ds.time.dt.time
sub  = select_times(ds['momo.t'], 1, time)
#%%
sub


sub.coords['time'] = sub.time.dt.floor('1D')

sub.groupby('time').mean()

#%%
import xarray as xr

ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias-24hour/jan/rf/2012/test.data.nc', engine='netcdf4')
ds
ds.close()
help(xr.open_dataarray)


#%%

from glob import glob

files_momo = glob.glob(f'{dirs}/rf/*/test.data.nc')

#%%
import os
import xarray as xr

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
years = ['2011', '2012', '2013', '2014', '2015']

for month in months:
    for year in years:
        v1 = f'/data/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias-8hour/{month}/rf/{year}/test.predict.nc'
        v2 = f'/data/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias-8hour-v2/{month}/rf/{year}/test.predict.nc'
        if os.path.exists(v1) and os.path.exists(v2):
            v1 = xr.open_dataset(v1).load()
            v2 = xr.open_dataset(v2).load()
            if v1.identical(v2):
                print(f'Good | {month}/{year} |')
            else:
                print(f'     | {month}/{year} | Bad')

# Results:
# train.target: Good
# test.targt  : Good
# train.data  : Good
# test.data   : Good
# test.predict: Bad
# Reason: Random state was not being set
#%%
