#%%
"""
Slashes (/) cannot be used in variable names due to NetCDF4 using HDF5 as the backend file writer. NetCDF4 is simply an API on top.
HDF5 uses slashes to denote levels in the file. Calling `xarray.to_netcdf` will raise exceptions as a result. Using `engine='scipy'`
can resolve and allow slashes in names, but scipy uses NetCDF3. Unfortunately, NetCDF3 does not support variable length string dtypes
(S).
"""
#%%

import xarray as xr
from glob import glob
from tqdm import tqdm
import os


def momo():
    files = glob('/data/MLIA_active_data/data_SUDSAQ/data/momo/**/*.nc')

    for file in tqdm(files, desc='Processing MOMO'):
        ds = xr.open_dataset(file, mode='r', engine='scipy')
        ds.load()

        # Rename the variables for THIS file
        names = {}
        for name in ds:
            names[name] = f"momo.{name.replace('/', '.')}"

        ds = ds.rename(names)

        os.remove(file)

        ds.to_netcdf(file, engine='netcdf4')

        del ds

def toar():
    files = glob('/data/MLIA_active_data/data_SUDSAQ/data/toar/matched/**/*.nc')

    for file in tqdm(files, desc='Processing TOAR'):
        ds = xr.open_dataset(file, mode='r', engine='scipy')
        ds.load()

        # Rename the variables for THIS file
        names = {}
        for name in ds:
            names[name] = name.replace('/', '.')

        ds = ds.rename(names)

        os.remove(file)

        ds.to_netcdf(file, engine='netcdf4')

        del ds

def fix_toar():
    files = glob('/data/MLIA_active_data/data_SUDSAQ/data/toar/matched/**/*.nc')
    for file in tqdm(files, desc='Processing TOAR'):
        ds = xr.open_dataset(file, mode='r', engine='netcdf4')
        ds.load()
        names = {}
        for name in ds:
            names[name] = name.replace('momo.', '')
        ds = ds.rename(names)
        os.remove(file)
        ds.to_netcdf(file, engine='netcdf4')
        del ds
