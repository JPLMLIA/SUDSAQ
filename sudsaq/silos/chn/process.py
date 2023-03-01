"""
"""
import argparse
import logging
import xarray as xr

from sudsaq import  Config
from sudsaq.data   import save_by_month
from sudsaq.utils  import init

Logger = logging.getLogger('sudsaq/silos/chn/process.py')

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
    import pandas as pd

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

def process():
    """
    """
    # Retrieve the config
    config = Config()

    # Load data in
    Logger.info(f'Loading {config.input}')
    ds = xr.open_mfdataset(config.input, parallel=True)
    ds.load()

    # Rename variables to be more SUDSAQ thematic
    ds = ds.rename_vars({
        'O3'           : 'o3',
        'site_city'    : 'city',
        'site_province': 'province',
        'site_lon'     : 'lon',
        'site_lat'     : 'lat'
    })

    # Calculate MDA8
    Logger.info('Calculating MDA8 on O3')
    ds['o3.mda8'] = ds['o3'].resample(time='1D').apply(mda)
    # ds['o3.mda8'] = ds['o3'].rolling(time=8).mean().resample(time='1D').max()

    # Save output
    if config.output.by_month:
        save_by_month(ds, config.output.path)
    else:
        Logger.info(f'Saving to output: {config.output}')
        ds.to_netcdf(config.output, engine='netcdf4')

    return True


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
