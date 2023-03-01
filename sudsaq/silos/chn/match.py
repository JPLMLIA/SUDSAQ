"""
"""
import argparse
import logging
import xarray as xr

from sudsaq import Config
from sudsaq.silos.match import match
from sudsaq.utils       import init

Logger = logging.getLogger('sudsaq/silos/chn/match.py')

def chn_match():
    """
    """
    # Retrieve the config
    config = Config()

    # Load in toar and momo data
    Logger.info(f'Loading MOMO')
    ds = xr.open_mfdataset(config.input.momo.regex, engine='scipy', parallel=True)

    Logger.info(f'Loading CHN ({config.input.toar.parameter})')
    cs = xr.open_mfdataset(config.input.chn.regex, engine='scipy', parallel=True)

    # Run the generalized matching function
    Logger.info('Matching CHN with MOMO')
    return match(ds, cs, f'chn/{config.metric}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'match',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())

    state = False
    try:
        state = chn_match()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
