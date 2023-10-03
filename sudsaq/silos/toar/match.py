"""
"""
import argparse
import logging
import pandas as pd
import xarray as xr

from sudsaq import Config
from sudsaq.silos.match import match
from sudsaq.utils       import init

Logger = logging.getLogger('sudsaq/silos/toar/match.py')

def toar_match():
    """
    """
    # Retrieve the config
    config = Config()

    # Load in toar and momo data
    Logger.info(f'Loading MOMO')
    ds = xr.open_mfdataset(config.input.momo.regex, parallel=False)
    ds = ds.sortby('lon')

    Logger.info(f'Loading TOAR ({config.input.toar.parameter})')
    df = pd.read_hdf(config.input.toar.file, config.input.toar.parameter)

    if 'station_lon' in df:
        # Rename to generic names
        df = df.rename(columns={
            'station_lon': 'lon',
            'station_lat': 'lat'
        })
    # # Convert from (-180, 180) to (0, 360) longitude format
    # df.lon.loc[df.lon < 0] += 360
    df    = df.rename(columns={'time': 'date'})
    dates = pd.unique(df['date'])

    # Run the generalized matching function
    Logger.info('Matching TOAR with MOMO')
    return match(ds, df, f'toar.{config.input.toar.parameter}', dates=dates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-i', '--inherit',  nargs    = '?',
                                            metavar  = 'sect1 sect2',
                                            help     = 'Order of keys to apply inheritance where rightmost takes precedence over left'
    )

    init(parser.parse_args())

    state = False
    try:
        state = toar_match()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
