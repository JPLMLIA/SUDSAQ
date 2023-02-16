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
