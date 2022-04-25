#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import argparse
import json
import logging
import pandas as pd

from glob import glob
from tqdm import tqdm

from sudsaq.config import Config

#%%

dfs   = []
bad   = []
files = glob('local/data/dev/**/**/**/*.json')
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

df = pd.concat(dfs)
df = df.set_index(['network', 'station', 'datetime'])

df

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
