#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import xarray as xr

#%%
ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar
ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc')

#%%

ds

md = xr.open_dataset('local/data/dev/toar/metadata.test.nc')
md

#%%

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc',
    '/Users/jamesmo/projects/suds-air-quality/local/data/dev/toar/metadata.test.nc'
]

ds = xr.open_mfdataset(files, parallel=True)

ds.load()

#%%
data = ds.stack({'loc': ['lat', 'lon', 'time']})
data

#%%
data['toar.station_alt.std'].mean()
md['toar.station_alt.std'].mean()

data.isel(loc=50600)

md.isel(lat=50, lon=50)

md['toar.station_alt.mean'].where(~md['toar.station_alt.mean'].isnull())
