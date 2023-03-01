#%%
#%%
#%%
import numpy as np

from glob import glob
from tqdm import tqdm

from sudsaq.utils import load_pkl

#%%

base = '/Volumes/MLIA_active_data/data_SUDSAQ/models/bias'
base = '/data/MLIA_active_data/data_SUDSAQ/models/bias'
paths = glob(f'{base}/**/[0-9]*', recursive=True)

#%%

runs = set()
for path in paths:
    try:
        split = path.split('/')
        int(split[-1])
        runs.update(['/'.join(split[:-2])])
    except:
        pass

#%%

results = {}
for run in tqdm(runs, desc='Runs Processed', position=1):
    files = glob(f'{run}/**/**/model.pkl')
    means = []
    for file in tqdm(files, desc='Models Processed', position=0):
        model = load_pkl(file)
        means.append(
            np.mean([est.get_depth() for est in model.estimators_])
        )
    results[run] = np.mean(means)
    print(f'{run} = {results[run]}')

# BIAS                   Run Path = Average model depth
models/bias/utc/8hr_median/v1     = 52.42533333333334
models/bias/local/8hr_median/v1   = 52.25033333333332
models/bias/local/8hr_median/v2   = 52.186
models/bias/local/8hr_median/v3   = 51.806000000000004
models/bias/local/8hr_median/v4   = 52.01666666666668
models/bias/local/8hr_median/v4.1 = 51.8825
models/bias/local/8hr_median/v5   = 44.143
# TOAR
models/toar/local/mean/v4         = 46.142
models/toar/local/mean/v5         = 40.71533333333333
# EMULATOR
models/emulator/utc/mda8/toar-limited/v4 = 41.299499999999995
