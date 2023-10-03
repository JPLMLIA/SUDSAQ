import numpy as np

from glob import glob
from tqdm import tqdm

from sudsaq.utils import (
    load_pkl,
    save_pkl
)


exclude = ['2011-2015', 'emulator/utc/mda8/smalls/full/v4']


def gather_stats(base='/data/MLIA_active_data/data_SUDSAQ/models/'):
    """
    """
    print(f'Discovering possible runs from {base}')
    paths = glob(f'{base}/**/[0-9]*', recursive=True)
    print(f'Found {len(paths)} paths')

    for ex in exclude:
        print(f'Excluding: {ex}')
        orig  = len(paths)
        paths = [path for path in paths if ex not in path]
        print(f'Paths reduced by {orig - len(paths)}, now: {len(paths)}')

    runs = set()
    for path in paths:
        try:
            split = path.split('/')
            int(split[-1])
            runs.update(['/'.join(split[:-2])])
        except:
            pass

    print(f'Discovered {len(runs)} runs from these paths')

    stats = {}
    for run in tqdm(runs, desc='Runs Processed', position=1):
        name = run[len(base):]
        tqdm.write(f'Processing run: {name}')
        stats[name] = data = {
            'params': None,
            'depth' : [],
            'leaves': [],
            'avg'   : {
                'depth' : 0,
                'leaves': 0
            }
        }
        files = glob(f'{run}/**/**/model.pkl')
        for file in tqdm(files, desc='Models Processed', position=0):
            try:
                model = load_pkl(file)
                data['params'] = model.get_params()
                for est in model.estimators_:
                    data['depth'].append(est.get_depth())
                    data['leaves'].append(est.get_n_leaves())
                del model
            except Exception as e:
                tqdm.write(f'Failed to load {file}:\n{e}')

        data['avg']['depth']  = np.mean(data['depth'])
        data['avg']['leaves'] = np.mean(data['leaves'])

    save_pkl(stats, 'stats.pkl')

gather_stats()

#%%

stats = load_pkl('/Volumes/MLIA_active_data/data_SUDSAQ/models/stats.pkl')

#%%

from sudsaq.utils import align_print

depths = {run: data['avg']['depth'] for run, data in stats.items()}

# Sort by value
depths = {k: depths[k]
    for k in sorted(depths, key=lambda k: depths[k])
}

print('Param: max_depth')
print(f'Average Depth of All Runs: {np.mean(list(depths.values()))}')

_ = align_print(depths, prepend='  ')

#%%

import pandas as pd


params = ['max_depth', 'max_leaf_nodes', 'max_features', 'n_estimators']
# params = list(stats[list(stats)[0]]['params'].keys())
kinds  = ['Average', 'Param']
items  = ['depth', 'leaves']
calc   = ['leaves/depth']
cols   = pd.MultiIndex.from_tuples(
    list(product(['Average'], items))
    + list(product(['Stat'], calc))
    + list(product(['Parameter'], params))
)
ind = ['Average Over All'] + sorted(stats.keys())
df  = pd.DataFrame(columns=cols, index=ind)


for run, data in stats.items():
    run = df.loc[run]

    for item in items:
        run['Average', item] = np.mean(data['avg'][item])

    for param in params:
        run['Parameter', param] = data['params'][param]

mean = df.mean()
df.iloc[0].loc[mean.index] = mean

df['Stat', 'leaves/depth'] = df['Average', 'leaves'] / df['Average', 'depth']

df

df.sort_values(('Average', 'leaves'))
#%%

df

#%%
%matplotlib inline

#%%

from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(15, 10))

#%%
class c:
    ad = ('Average', 'depth')
    al = ('Average', 'leaves')

baseline = 'emulator/utc/mda8/smalls/toar-limited/v4'

fig, ax = plt.subplots(figsize=(10, 5))
plot = lambda df, **kwargs: df.plot(ax=ax, rot=90, kind='bar', **kwargs)

sf = df.sort_values(c.al)
sf[c.al]

# plot(sf[[('Average', 'leaves'), ('Average', 'depth')]])


sf[c.al] /= sf[c.al][baseline]
sf[c.ad]  /= sf[c.ad][baseline]

plot(sf[[c.al, c.ad]], title=f'\nbase={baseline}')
