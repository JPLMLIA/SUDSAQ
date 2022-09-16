#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import xarray as xr

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2009/01.nc',
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
for offset, (west, east) in timezones:
    sub  = ds.sel(lon=slice(west, east))
    time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time
    mask = ( dt.time(8) < time) & (time < dt.time(16) )
    data.append(sub.where(mask, drop=True).resample(time='1D').mean())

ns = xr.merge(data)#, compat='override')
ns

#%%
ns.load() # override: 5.560 + 20.675
# default: 17.311 + 2.774
ns
