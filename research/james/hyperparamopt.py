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
