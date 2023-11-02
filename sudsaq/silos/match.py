"""
"""
import logging
import numpy  as np
import pandas as pd
import xarray as xr

from scipy import stats
from tqdm  import tqdm

from mlky import Config

from sudsaq.data import save_by_month

Logger = logging.getLogger('sudsaq/silos/match.py')

def match(ds, df, tag, dates=None):
    """
    Matches silo data to the MOMO lat/lon grid using silo lat/lon data.

    Parameters
    ----------
    ds: xarray.Dataset
        The MOMO Dataset object containing the dimensions latitude and longitude
        to be used for matching
    df: pandas.DataFrame
        Silo dataframe in expected format. Must contain columns: [time, lat, lon]
    tag: str
        The unique tag for this silo. For example, TOAR is `toar.{Config.input.toar.variable}`

    Returns
    -------
    bool or int
        Returns True if the function finished properly, or an int indicating the status code.
        See notes for more information on status codes.

    Notes
    -----
    Status Codes:
    1 - An invalid metric to be used in scipy.stats.binned_statistic_2d was provided
    """
    # Validate the metrics requested
    Logger.info('Validating Config metrics')
    valid = ['mean', 'std', 'median', 'count', 'sum', 'min', 'max']
    for metric in Config.metrics:
        if metric not in valid:
            Logger.error(f'Metric {metric!r} is not in the list of valid options: {valid}')
            return 1

    # Collect the dates to process
    if not dates:
        Logger.info('Retrieving unique dates')
        dates = pd.unique(df.index.get_level_values('date')) # TODO: toar/match.py should handle the index to make this script more generalized
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
    shape = *dates.shape, *ds.lat.shape, *ds.lon.shape
    Logger.debug('Creating the matched dataset')
    Logger.debug(f'Shape: {shape}')
    for metric in tqdm(Config.metrics, 'Allocating memory'):
        ms[f'{tag}.{metric}'] = (('time', 'lat', 'lon'), np.full(shape, np.nan))

    # Add end values for the last bin of each lat/lon
    lat = np.hstack([ds['lat'],  90.])
    lon = np.hstack([-180., ds['lon']])

    Logger.debug(f'Lat: {lat}')
    Logger.debug(f'Lon: {lon}')

    # # For each date, compute the metrics and save into the Dataset
    for time in tqdm(dates, desc='Processing Dates'):
        tf = df.query('date == @time')

        for metric in Config.metrics:
            calc = stats.binned_statistic_2d(
                tf.lat,
                tf.lon,
                tf[Config.variable],
                metric,
                bins = [lat, lon],
                expand_binnumbers = True
            )
            # Now save the calculation for this time
            ms.loc[{'time': time}][f'{tag}.{metric}'][:] = calc.statistic

    # for metric in Config.metrics:
    #     calc = stats.binned_statistic_2d(
    #         df.lat,
    #         df.lon,
    #         df[Config.variable],
    #         metric,
    #         bins = [lat, lon],
    #         expand_binnumbers = True
    #     )
    #     # Now save the calculation for this time
    #     print(calc.statistic.shape)
    #     print()
    #     ms[f'{tag}.{metric}'][:] = calc.statistic

    ms['time'] = pd.DatetimeIndex(ms['time'])
    Logger.debug(f'Dataset:\n{ms}')

    # Save output
    if Config.output.by_month:
        save_by_month(ms, Config.output.path)
    else:
        Logger.info(f'Saving to output: {Config.output.path}')
        ms.to_netcdf(Config.output.path)

    return True
