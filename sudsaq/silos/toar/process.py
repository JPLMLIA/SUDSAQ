#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import argparse
import json
import logging
import numpy  as np
import pandas as pd
import xarray as xr

from glob import glob
from tqdm import tqdm

from sudsaq.config import Config

#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'process',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())
    Logger = logging.getLogger('sudsaq/silos/toar/process.py')

    state = False
    try:
        state = process()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')

#%%
#%%
#%%
#%%

glob('/Volumes/MLIA_active_data/data_SUDSAQ/processed/coregistered/*')

#%%

glob('/Volumes/MLIA_active_data/data_SUDSAQ/MOMO/outputs/*')

glob('/Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/*')

!ls -lah /Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/

#%%

import h5py

def list_h5(h5, details=False, _t=1):
    if isinstance(h5, str):
        h5 = h5py.File(h5, 'r')
    ret = h5.filename.split('/')[-1] if _t == 1 else ''
    for key in h5:
        if not isinstance(h5[key], h5py.Dataset):
            ret += f"\n{'|'*_t}> {key}{list_h5(h5[key], details, _t+1)}"
        else:
            ret += f"\n{'|'*_t}- {key}"
            if details:
                ret += f' (Size: {h5[key].size}, dtype: {h5[key].dtype})'
    if _t == 1:
        print(ret)
        h5.close()
    else:
        return ret

# list_h5('/Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/momo_2012_01.h5')
list_h5('/Volumes/MLIA_active_data/data_SUDSAQ/toar2.v1.variables.h5')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

momo_file = f'{momo_output}/momo_{year}_{month}.h5'
momo = {}
with closing(h5py.File(momo_file, 'r')) as f:
    keys = list(f)
    for k in keys:
        momo[k] = f[k][:]

# momo = read_data_momo.main(output_var = names[0], input_var = names[1,2],
#                 year = year, month = month)
momo_lon = np.hstack([momo['lon'], 360.]) - 180
momo_lat = np.hstack([momo['lat'], 90.])
x, y = np.meshgrid(momo['lon'], momo['lat'])

#%%
#%%

# toar data
file = '/Users/jamesmo/projects/suds-air-quality/local/data/toar2.v1.variables.h5'
df = pd.read_hdf(file, 'o3')

#%%
import h5py
import numpy as np

momo_files = glob('/Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/*')
momo_files = momo_files[:1]
momo_files = ['/Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/momo_2016_02.h5']
for file in momo_files:
    momo = {}
    with h5py.File(file, 'r') as h5:
        shape = (
            h5['date'].shape[0],
            h5['lat'].shape[0],
            h5['lon'].shape[0]
        )
        for key in h5.keys():
            momo[key] = h5[key][:]

#%%
dates = pd.DatetimeIndex(['-'.join(date) for date in momo['date'].astype(str)])
ds = xr.Dataset(
    {
        key: (('date', 'lat', 'lon'), data)
        for key, data in momo.items() if data.shape == shape
    },
    coords = {
        'date': dates,
        'lat': momo['lat'],
        'lon': momo['lon'] - 180
    }
)

#%%
def load_momo(file):
    """
    """
    # Load momo in raw
    momo = {}
    with h5py.File(file, 'r') as h5:
        shape = (
            h5['date'].shape[0],
            h5['lat'].shape[0],
            h5['lon'].shape[0]
        )
        for key in h5.keys():
            momo[key] = h5[key][:]

    # Convert to datetime format
    dates = pd.DatetimeIndex(['-'.join(date) for date in momo['date'].astype(str)])

    # Now convert momo data to xarray
    ds = xr.Dataset(
        {
            key: (('date', 'lat', 'lon'), data)
            for key, data in momo.items() if data.shape == shape
        },
        coords = {
            'date': dates,
            'lat': momo['lat'],
            'lon': momo['lon'] - 180
        }
    )

    return ds

def match(ds, df):
    """
    """
    # Prepare the momo lat, lon for stats.binned_statistic_2d
    lon = np.hstack([ds['lon'], 180.])
    lat = np.hstack([ds['lat'], 90.])

    # Metrics to extract with stats.binned_statistic_2d
    metrics = ['mean', 'std', 'count']

    # Setup the metric variables
    shape = (*ds.date.shape, *ds.lat.shape, *ds.lon.shape)
    for metric in metrics:
        ds[f'toar/{metric}'] = (('date', 'lat', 'lon'), np.full(shape, np.nan))

    # Easy date formatting function to make it work with DataFrame.query
    format = lambda date: str(date.values).replace('-', '_').split('T')[0]

    # Select on this month's data
    nf = df.query(f"{format(ds.date[0])} <= date <= {format(ds.date[-1])}")

    # Collect the dates
    dates = pd.unique(nf.index.get_level_values('date'))

    # Process per date
    for date in dates:
        tf = nf.query('date == @date')

        for metric in metrics:
            calc = stats.binned_statistic_2d(
                tf.station_lat,
                tf.station_lon,
                tf.dma8epa,
                metric,
                bins = [lat, lon],
                expand_binnumbers = True
            )
            # Now save the calculation for this date
            ds.loc[{'date': date}][f'toar/{metric}'][:] = calc.statistic

    return ds

dss = []
files = glob('/Volumes/MLIA_active_data/data_SUDSAQ/processed/summary_dp/MOMO/*')
for file in tqdm(files, desc='MOMO Files Processed'):
    ds = load_momo(file)
    ds = match(ds, df)
    dss.append(ds)

#%%
def geospatial(data, title=None):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    # ax.set_extent([-20, 20, 30, 60], ccrs.PlateCarree())
    data.plot.pcolormesh(x='lon', y='lat', ax=ax, levels=10, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)
    if title:
        ax.set_title(title)
    plt.show()

#%%
geospatial(ds['toar/mean'].mean('date'))

#%%
!ls -lah local/data/toar2.v1.matched.nc
ds = xr.concat(dss, 'date')
ds.to_netcdf('local/data/toar2.v1.matched.nc', engine='scipy')

ds['toar/bias'] = ds['o3'] - ds['toar/mean']
ds
#%%

for year, yds in ds.groupby('date.year'):
    vmin = yds['toar/mean'].quantile(.01)
    vmax = yds['toar/mean'].quantile(.99)
    for season, sds in yds.groupby('date.season'):
        geospatial(sds['toar/mean'].mean('date'), title=f'{season} {year}')
    break
