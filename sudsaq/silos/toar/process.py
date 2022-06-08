#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import json
import pandas as pd

from glob import glob
from tqdm import tqdm


dfs   = []
bad   = []
files = glob('./v1/**/**/**/*.json')
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

df.to_hdf('data.h5', 'v1')

#%%
import matplotlib.pyplot as plt
import seaborn as sns

nf = df.reset_index()
nf.index = pd.to_datetime(nf.datetime)

gf = nf.groupby(pd.Grouper(freq='M'))
gf = gf.count()
gf = gf.sum(axis=1)
ax = gf.plot(kind='bar', figsize=(50, 10))

plt.tight_layout()
plt.savefig('data_counts_by_month.png')

#%%
