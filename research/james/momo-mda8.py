#%%
#%%
#%%
import pandas as pd
import xarray as xr

from glob import glob
from tqdm import tqdm


def mda(data, offset=8, inclusive=False):
    """
    Calculates the maximum daily average for a variable using a time slice.
    Does NOT include times from the next day.

    Parameters
    ----------
    data: xarray.DataArray
        A single day of data with a `time` dimension
    offset: int, default=8
        Hour many hours to average
    inclusive: bool, default=False
        Whether the offset timestamp is to be included if it lands on a valid timestamp

    Returns
    -------
    xarray.DataArray
        Maximum [offset]-hour daily average
    """
    # This is the offset value to add to each t for slicing
    delta = pd.Timedelta(hours=offset)
    if not inclusive:
        delta -= pd.Timedelta(1)

    # For each timestamp, find the MDA
    size = None
    mda  = []
    for ts in data.time:
        # Select between t and t+n
        sel = data.sel(time=slice(ts, ts+delta))

        # The first selection sets the expected size (prevents end selection issues)
        if not size:
            size = sel.time.size

        if sel.time.size == size:
            mda.append(sel.mean('time'))
        else:
            break

    # Merge these results together
    mda = xr.concat(mda, 'mda')

    # Return the max
    return mda.max('mda')


files = glob('fixed/**/*.nc')

for file in tqdm(files):
    print(f'Calculating mda8 for: {file}')

    with xr.open_dataset(file) as ds:
        coords = ds.coords
        mda8 = ds['momo.o3'].load().resample(time='1D').apply(mda)

    mda8['time'] = mda8['time'] + pd.Timedelta(hours=1)
    mda8 = mda8.to_dataset(name='sudsaq.o3.mda8').reindex(coords)
    mda8.to_netcdf(file, mode='a')

#%%

ds = xr.open_dataset(files[0])
ms = ds['sudsaq.o3.mda8'].load()
ms.isel(time=48)

#%%

ds = ds[['momo.mda8', 'sudsaq.o3.mda8']]
ds.load()


ds['diff'] = ds['momo.mda8'] - ds['sudsaq.o3.mda8']
ds.isel(time=24)

ds.time.size
sel = ds.isel(time=list(range(0, ds.time.size, 12)))
sel.mean(['lat', 'lon'])

#%%

import plotly.express as px

mean = sel.mean(['lat', 'lon'])

px.line(x=mean['time'], y=mean['diff'])

#%%
import xarray as xr

from glob import glob
from tqdm import tqdm

files = glob('fixed/**/*.nc')

working = '2005'
dss = []
for file in tqdm(files):
    year = file.split('/')[-2]

    if year != working:
        ds = xr.merge(dss)
        ds.to_netcdf(f'{working}.mda8s.nc')
        dss = []
        del ds
        working = year

    print(f'Calculating mda8 for: {file}')
    with xr.open_dataset(file) as ds:
        ns = ds[['momo.mda8', 'sudsaq.o3.mda8']].load()

    ns['diff'] = ns['momo.mda8'] - ns['sudsaq.o3.mda8']
    dss.append(ns)

#%%
#%%
#%%

mean['momo.mda8'][1] = float('nan')
mean['momo.mda8'] + mean['sudsaq.o3.mda8']
import numpy as np
np.nansum([mean['momo.mda8'], mean['sudsaq.o3.mda8']])

xr.merge(mean['momo.mda8'], mean['sudsaq.o3.mda8'])

c = mean['momo.mda8'].combine_first(mean['sudsaq.o3.mda8'])
c
mean

help(mean.combine_first)

#%%
import re

def calc(ds, string):
    for key in list(ds):
        # Find this key not followed by a digit or word character (eg. prevents momo.no matching to momo.no2)
        string = re.sub(fr'({key})(?!\d|\w)', f"ds['{key}']", string)

    print(f'Attempting to evaluate: {string!r}')
    return eval(string)

calc(ds, "momo.mda8 - sudsaq.o3.mda8")
calc(mean, "(momo.mda8).combine_first(sudsaq.o3.mda8)")

#%%
ds

#%%

from mlky import Config, Sect

Config('sudsaq/configs/definitions.yml', 'default<-mac<-v4<-v6r<-bias-median<-toar-v2.3<-RFQ<-extended-no-france-2014<-jan')
Config.input.glob
Config.input.glob[0]

s = Sect('sudsaq/configs/definitions.yml')
s['extended-no-france-2014'].input

#%%

import plotly.express as px
import xarray as xr

ds = xr.open_mfdataset('.local/data/momo/mda8s/*.nc').load()
ms = ds.isel(time=list(range(0, ds.time.size, 12)))

#%%
lines = []
for year, yds in ms.groupby('time.year'):
    mean = yds.mean(['lat', 'lon'])
    line = px.line(x=mean['time'], y=mean['diff'])
    break

#%%

years = []
means = ms.mean(['lat', 'lon'])
for year, yds in means.groupby('time.year'):
    df = yds.to_dataframe()
    if f'{year}-02-29 01:00:00' in df.index:
        df = df.drop(index=[f'{year}-02-29 01:00:00'])
    df = df.reset_index()
    df['time'] = year
    df.index = pd.date_range(start='2001', periods=365, freq='D')
    years.append(df)

#%%
years
#%%

df = pd.concat(years)
df.index.name = 'Date'
px.line(df, y='diff', color='time', title='momo.mda8 - sudsaq.o3.mda8')

#%%

pd.to_datetime(2001 * 1000 + df.index + 1, format='%Y%j')
df.index


2000 * 1000 + df.index

df
