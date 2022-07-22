#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
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

#%%
config = Config('sudsaq/configs/dev/ml/dev.yml', 'create-rf')
model  = load_pkl('local/runs/1.create-rf/fold_0/model.pkl')

data   = xr.open_dataarray('local/runs/1.create-rf/fold_0/test.data.nc')
data   = data.stack(loc=['lat', 'lon', 'time'])
data   = data.transpose('loc', 'variable')
data   = data.dropna('loc')

target = xr.open_dataarray('local/runs/1.create-rf/fold_0/test.target.nc')
target = target.stack(loc=['lat', 'lon', 'time'])
target = target.dropna('loc')

predict = xr.open_dataarray('local/runs/1.create-rf/fold_0/test.predict.nc')
predict = predict.stack(loc=['lat', 'lon', 'time'])
predict = predict.dropna('loc')

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
scores = align_print(stats)
