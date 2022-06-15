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
    Logger.info('Validating config metrics')
    valid = ['mean', 'std', 'median', 'count', 'sum', 'min', 'max']
    for metric in config.metrics:
        if metric not in valid:
            Logger.error(f'Metric {metric!r} is not in the list of valid options: {valid}')
            return 1

    # Collect the dates to process
    Logger.info('Retrieving unique dates')
    dates = pd.unique(df.index.get_level_values('date'))
    Logger.debug(f'Number of dates: {len(dates)}')

    # Prepare the Dataset that the matched data will reside in
    ms = xr.Dataset(
        coords = {
            'time': dates,
            'lat' : ds['lat'],
            'lon' : ds['lon']
        }
    )
    # Prefill the variables
    Logger.debug('Creating the matched dataset')
    shape = *dates.shape, *ds.lat.shape, *ds.lon.shape
    for metric in config.metrics:
        ms[f'{tag}/{config.metric}/{metric}'] = (('time', 'lat', 'lon'), np.full(shape, np.nan))

    # Add end values for the last bin of each lat/lon
    lat = np.hstack([ds['lat'], 90.])
    lon = np.hstack([ds['lon'], 360.])

    # For each date, compute the metrics and save into the Dataset
    for time in tqdm(dates, desc='Processing Dates'):
        tf = df.query('date == @time')

        for metric in config.metrics:
            calc = stats.binned_statistic_2d(
                tf.lat,
                tf.lon,
                tf[config.metric],
                metric,
                bins = [lat, lon],
                expand_binnumbers = True
            )
            # Now save the calculation for this time
            ms.loc[{'time': time}][f'{tag}/{config.metric}/{metric}'][:] = calc.statistic

    # Save output
    if config.output.by_month:
        Logger.info('Saving output by month')
        for year, yms in ms.groupby('time.year'):
            # Check if directory exists, otherwise create it
            output = f'{config.output.path}/{year}'
            if not os.path.exists(output):
                os.mkdir(output, mode=0o771)

            for month, mms in yms.groupby('time.month'):
                mms.to_netcdf(f'{output}/{month:02}.nc', engine='scipy')
    else:
        Logger.info(f'Saving to output: {config.output}')
        ms.to_netcdf(config.output, engine='scipy')

    return True
