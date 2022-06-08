"""
"""
import logging
import numpy  as np
import os
import pandas as pd
import xarray as xr

from scipy import stats
from tqdm  import tqdm

from sudsaq.config import Config

Logger = logging.getLogger('sudsaq/silos/match.py')

def match(ds, df, tag):
    """
    """
    # Retrieve the config
    config = Config()

    # Validate the metrics requested
    valid = ['mean', 'std', 'median', 'count', 'sum', 'min', 'max']
    for metric in config.metrics:
        if metric not in valid:
            Logger.error(f'Metric {metric!r} is not in the list of valid options: {valid}')
            return 1

    # Collect the dates to process
    dates = pd.unique(df.index.get_level_values('date'))

    # Prepare the Dataset that the matched data will reside in
    ts = xr.Dataset(
        coords = {
            'time': dates,
            'lat' : ds['lat'],
            'lon' : ds['lon']
        }
    )
    # Prefill the variables
    shape = *dates.shape, *ds.lat.shape, *ds.lon.shape
    for metric in config.metrics:
        ts[f'{tag}/{config.metric}/{metric}'] = (('time', 'lat', 'lon'), np.full(shape, np.nan))

    # Add end values for the last bin of each lat/lon
    lat = np.hstack([ds['lat'], 90.])
    lon = np.hstack([ds['lon'], 360.])

    # For each date, compute the metrics and save into the Dataset
    for time in tqdm(dates, desc='Processing Dates'):
        tf = df.query('date == @time')

        for metric in config.metrics:
            calc = stats.binned_statistic_2d(
                tf.station_lat,
                tf.station_lon,
                tf[config.metric],
                metric,
                bins = [lat, lon],
                expand_binnumbers = True
            )
            # Now save the calculation for this time
            ts.loc[{'time': time}][f'{tag}/{config.metric}/{metric}'][:] = calc.statistic

    # Save output
    if config.output.by_month:
        for year, yts in ts.groupby('time.year'):
            # Check if directory exists, otherwise create it
            output = f'{config.output.path}/{year}'
            if not os.path.exists(output):
                os.mkdir(output, mode=0o771)

            for month, mts in yts.groupby('time.month'):
                mts.to_netcdf(f'{output}/{month:02}.nc', engine='scipy')
    else:
        ts.to_netcdf(config.output, engine='scipy')

    return True

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
# geospatial(ds['toar/mean'].mean('date'))
