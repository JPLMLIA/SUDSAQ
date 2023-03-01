import argparse
import matplotlib.pyplot as plt
import os
import xarray as xr

from glob import glob
from tqdm import tqdm

from sudsaq import Config
from sudsaq.utils      import load_pkl
from sudsaq.ml.analyze import analyze

def correct(dir):
    rm = glob(f'{dir}/**/rf/**/*.bias.nc') + glob(f'{dir}/**/rf/**/*.contributions.nc')
    for file in rm:
        os.remove(file)

    [yaml] = glob(f'{dir}/*.yml')

    paths = glob(f'{dir}/**/rf/*/')
    for path in tqdm(paths, desc='Models Corrected'):
        month, _, year = path.split('/')[-4:-1]
        print(f'Processing: {month}/{year}')
        config = Config(yaml, month)
        config.not_ti               = True
        config.output.data          = False
        config.output.target        = False
        config.output.bias          = False
        config.output.contributions = False
        config.output.model         = False
        if os.path.exists(f'{path}/test.data.nc'):
            data   = xr.open_dataset(f'{path}/test.data.nc').to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
            target = xr.open_dataarray(f'{path}/test.target.nc').stack(loc=['lat', 'lon', 'time']).dropna('loc')
            model  = load_pkl(f'{path}/model.pkl')
            analyze(model=model, data=data, target=target, kind='test', output=path)
            plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-r', '--run',      type     = str,
                                            required = True,
                                            help     = 'Run directory to correct'
    )
    args = parser.parse_args()
    correct(args.run)
paths = glob(f'{dir}/**/rf/*/')
