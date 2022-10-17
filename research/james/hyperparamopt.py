#%%


#%%

from sudsaq.config import Config
from sudsaq.data import load

config = Config('research/james/create.bias.11-15.8hour_avg.yml', 'dec')

#%%

data, target = load(config, split=True, lazy=True)

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold
)

kfold  = GroupKFold(n_splits=len(set(t.time.dt.year.values)))
groups = t.time.dt.year.values

#%%
gscv = GridSearchCV(
    RandomForestRegressor(),
    cv          = kfold,
    error_score = 'raise',
    n_jobs      = -1,
    param_grid = {**config.hyperoptimize.GridSearchCV.param_grid}
    # **config.hyperoptimize.GridSearchCV
)

#%%

gscv.fit(d, t, groups=groups)



#%%
target.load(), data.load()
d, t = xr.align(data.dropna('loc'), target.dropna('loc'))

kfold  = GroupKFold(n_splits=len(set(t.time.dt.year.values)))
groups = t.time.dt.year.values

#%%
from sudsaq.utils import align_print

align_print(gscv.best_params_, enum=False, prepend='  ')

#%%
# Create the predictor
model = lambda: RandomForestRegressor(**config.model.params, **gscv.best_params_)

#%%

for fold, (train, test) in enumerate(kfold.split(data, target, groups=groups)):
    break

#%%

import xarray as xr

ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2011/12.nc')
ds = ds[['momo.o3', 'momo.t']]
ds.load()

#%%


import numpy as np

# Jan
original = {
    '2011': 9.01781726498881,
    '2012': 6.665053477876889,
    '2013': 7.668583078524768,
    '2014': 7.4542908970848325,
    '2015': 7.25707384958671,
    'mean': 7.612563713612403
}
new = {
    '2011': 10.82588738417951,
    '2012': 8.859429087127586,
    '2013': 9.437811915403447,
    '2014': 8.392965517166807,
    '2015': 8.762334472896613,
    'mean': 9.255685675354792
}
yuliya = {
    '2011': 9.16702664837093,
    '2012': 6.802099833716384,
    '2013': 7.8204676356869385,
    '2014': 7.6052759543272135,
    '2015': 7.370981639126759,
    'mean': 7.753170342245646
}

#%% July
original = {
    '2011': 10.951878347231903,
    '2012': 11.604376684823668,
    '2013': 14.116871420205392,
    '2014': 11.936351776675746,
    '2015': 12.770619920179662,
    'mean': 12.276019629823274
}
v2 = {
    '2011': 12.249954193630709,
    '2012': 11.977856785968422,
    '2013': 13.292692933216284,
    '2014': 12.950502819268113,
    '2015': 13.55053818385136,
    'mean': 12.804308983186976
}
new = {
    '2011': 12.414073485337394,
    '2012': 12.06181022972492,
    '2013': 12.120320174950669,
    '2014': 12.221558703137145,
    '2015': 12.592890595403517,
    'mean': 12.282130637710727
}
yuliya = {
    '2011': 9.57667454993185,
    '2012': 9.866878546182189,
    '2013': 10.117974835132852,
    '2014': 9.541064950046609,
    '2015': 9.82194274097163,
    'mean': 9.784907124453024
}




np.nanmean(list(yuliya.values()))

#%%
