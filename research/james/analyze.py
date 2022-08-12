#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%
import numpy  as np
import pandas as pd
import xarray as xr

from scipy.stats import (
    gaussian_kde,
    pearsonr
)
from sklearn.inspection import permutation_importance
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from treeinterpreter import treeinterpreter as ti
from sudsaq.ml       import treeinterpreter as aqti

from sudsaq.config import (
    Config,
    Section,
    Null
)
from sudsaq.data  import load, Dataset
from sudsaq.ml    import plots
from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)

import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sns

from sudsaq.config import Config

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

from types import SimpleNamespace

Logger = SimpleNamespace(
    exception = lambda string: print(f'EXCEPTION: {string}'),
    info      = lambda string: print(f'INFO: {string}'),
    error     = lambda string: print(f'ERROR: {string}'),
    debug     = lambda string: print(f'DEBUG: {string}')
)


config = Config('sudsaq/configs/dev/ml/dev.yml', 'nov')

#%%
model  = load_pkl('local/runs/11-14/bias/apr/rf/2013/model.pkl')

data   = xr.open_dataset('local/runs/11-14/bias/apr/rf/2013/test.data.nc')
# data   = data.stack(loc=['lat', 'lon', 'time'])
# data   = data.transpose('loc', 'variable')
# data   = data.dropna('loc')

target = xr.open_dataarray('local/runs/11-14/bias/apr/rf/2013/test.target.nc')
target = target.stack(loc=['lat', 'lon', 'time'])
target = target.dropna('loc')

predict = xr.open_dataarray('local/runs/11-14/bias/may/rf/2013/test.predict.nc')
predict = predict.stack(loc=['lat', 'lon', 'time'])
predict = predict.dropna('loc')

#%%



#%%
predict.close()
predict.reindex_like(ds[['lat', 'lon']])
#%%
#%%
#%%

def _ti():
    predict       = xr.zeros_like(target)
    bias          = xr.zeros_like(target)
    contributions = xr.zeros_like(data)

    predicts, bias[:], contributions[:] = ti.predict(model, data)
    predict[:] = predicts.flatten()

    return predict, bias, contributions

def _aq():
    predict       = xr.zeros_like(target)
    bias          = xr.zeros_like(target)
    contributions = xr.zeros_like(data)

    predicts, bias[:], contributions[:] = aqti.predict(model, data, n_jobs=-1)
    predict[:] = predicts.flatten()

    return predict, bias, contributions


#%%
from timeit import timeit

timeit(_ti, number=1, globals={'target': target, 'data': data}) # 216.22187531400004
timeit(_aq, number=1, globals={'target': target, 'data': data}) # 41.05707403399998

#%%

stats = Section('scores', {
    'mape'  : mean_absolute_percentage_error(target, predict),
    'rmse'  : mean_squared_error(target, predict, squared=False),
    'r2'    : r2_score(target, predict),
    'r corr': pearsonr(target, predict)[0]
})
scores = align_print(stats, enum=False, prepend='  ')

#%%

config.reset('feb')

data, target = load(config, split=True)

#%%
data.load().dropna('loc')
target.load().dropna('loc')

#%%
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold
)

kfold  = GroupKFold(n_splits=len(set(data.time.dt.year.values)))
groups = target.time.dt.year.values

for fold, (train, test) in enumerate(kfold.split(data, target, groups=groups)):
    input = Section('', {
        'data': {
            'train': data.isel(loc=train),
            'test' : data.isel(loc=test)
        },
        'target': {
            'train': target.isel(loc=train),
            'test' : target.isel(loc=test)
        }
    })
    d = input.data
    t = input.target

    print(f'fold_{fold} Is Finite ------------------------')
    print('Data   :')
    print(f' train: {np.isfinite(d.train).all().values}')
    print(f'  test: {np.isfinite(d.test).all().values}')
    print('Target :')
    print(f' train: {np.isfinite(t.train).all().values}')
    print(f'  test: {np.isfinite(t.test).all().values}')

    print(f'fold_{fold} Has NaN ------------------------')
    print('Data   :')
    print(f' train: {np.isnan(d.train).any().values}')
    print(f'  test: {np.isnan(d.test).any().values}')
    print('Target :')
    print(f' train: {np.isnan(t.train).any().values}')
    print(f'  test: {np.isnan(t.test).any().values}')

#%%

target = xr.open_dataarray('/Volumes/MLIA_active_data/data_SUDSAQ/runs/11-14/bias/jan/rf/fold_0/test.target.nc').load()
predict = xr.open_dataarray('/Volumes/MLIA_active_data/data_SUDSAQ/runs/11-14/bias/jan/rf/fold_0/test.predict.nc').load()
predict.load()

predict.plot()

#%%

fig, ax = plt.subplots(figsize=(20, 10))

ax.scatter(x)

#%%
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns

fig = plt.figure(figsize=(20, 10))
ax  = plt.subplot(111, projection=ccrs.PlateCarree())


#%%

# def parse_importances(file):

file = open('/Users/jamesmo/projects/suds-air-quality/local/runs/11-14/bias/jan/rf/fold_1/test.importance.txt', 'r')
lines = file.readlines()
lines

import re

#%%

r = re.findall(r'(?:^- \d: )(.*)(?: = )(.*)(?: \+/- )(.*)', lines[1])

#%%

features = {}
with open('/Users/jamesmo/projects/suds-air-quality/local/runs/11-14/bias/jan/rf/fold_0/test.importance.txt') as file:
    for line in file.readlines():
        if 'Feature importance' in line:
            kind = 'importance'
        elif 'Permutation importance' in line:
            kind = 'permutation'
        else:
            match = re.findall(r'(?:^- \d: )(.*)(?: = )(.*)(?: \+/- )(.*)', line)
            if match:
                feature, score, stddev = match[0]
                if feature not in features:
                    features[feature] = {'importance': {}, 'permutation': {}}
                features[feature][kind] = {
                    'score': score,
                    'std'  : stddev
                }

features

#%%

mimp = pd.read_hdf('/Users/jamesmo/projects/suds-air-quality/local/runs/11-14/bias/mar/rf/2012/test.importance.h5', 'model')
pimp = pd.read_hdf('/Users/jamesmo/projects/suds-air-quality/local/runs/11-14/bias/mar/rf/2012/test.importance.h5', 'permutation')

mimp

mimp.to_xarray()

ds = xr.Dataset(coords={'importance': ['model', 'permutation'], 'metric': ('importance', ['score', 'stddev'])})
ds['momo.t'] = (('importance', 'metric'), [[1, 1], [1, 1]])

#%%

d = data.unstack()
d.to_dataset()

d['variable'].unstack()

d.Variable.unstack()


d.unstack('variable')

#%%

ds = load(config, lazy=False)
ds.lat

#%%

t = target.unstack()

t
ds.isel(time=0)
_t = xr.zeros_like(ds)
xr.broadcast(t, ds, exclude='time')

help(xr.broadcast)

#%%
target
xr.align(t, ds, join='right')[0]

t
t.reindex_like(ds)

#%%

data.reindex_like(ds[['lat', 'lon']])


config.a != None


c = xr.open_dataarray('local/runs/11-14/bias/may/rf/2013/test.contributions.nc')
c

#%%

mi = pd.read_hdf('local/runs/11-14/bias/may/rf/2013/test.importance.h5', 'model')
pi = pd.read_hdf('local/runs/11-14/bias/may/rf/2013/test.importance.h5', 'permutation')

#%%

target
target.dropna('loc')
d = data.dropna('loc')

#%%


d.load()

d.drop(dim='variable')

target
#%%
d.isel(time=0)
xr.zeros_like(d.isel(variable=0).drop_vars('variable'))
d.drop_vars('variable')
z = xr.zeros_like(target)
z['loc']

#%%

dir(set)
help(set().intersection)

i = set(target['loc'].values).intersection(data['loc'].values)

data.sel(loc=i)

#%%

len(i)

#%%
data
d
target

#%%

_d, _t = xr.align(d, target)

_d

_t
#%%
la
_t

#%%
#%%
#%%

import numpy as np

lat = np.random.rand(10) * 10
lat.shape
lat

lat_mask = (lat > 5) & (lat < 8)
lon_mask =
mask = lat_mask & lon_mask

america = data[mask]

#%%

a = np.random.rand(3, 3)
a[[0, 1]]
