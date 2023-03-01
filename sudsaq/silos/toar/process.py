import argparse
import json
import logging
import pandas as pd

from glob import glob
from tqdm import tqdm

from sudsaq import  Config
from sudsaq.utils  import init

Logger = logging.getLogger('sudsaq/silos/toar/process.py')

def process():
    """
    """
    Logger.info(f'Processing JSON files')
    dfs   = []
    bad   = []
    files = glob(config.input.regex)
    for file in tqdm(files, desc='Loading JSONs'):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
        except:
            bad.append(file)
            continue

        if len(data) <= 2:
            continue

        metadata = data.pop('metadata')
        df = pd.DataFrame(data)

        df['network']     = metadata['network_name']
        df['station']     = metadata['station_id']
        df['station_lon'] = metadata['station_lon']
        df['station_lat'] = metadata['station_lat']

        dfs.append(df)

    Logger.info(f'Concatenating {len(dfs)} groups of data together')
    df = pd.concat(dfs)
    df = df.set_index(['network', 'station', 'datetime'])

    if bad:
        Logger.warning(f'There were {len(bad)} bad JSON files')

    Logger.info(f'Saving output to {config.output.file} under key {config.output.key}')
    df.to_hdf(config.output.file, config.output.key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'retrieve',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())

    state = False
    try:
        state = retrieve()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
