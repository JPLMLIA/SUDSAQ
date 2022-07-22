#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import numpy  as np
import pandas as pd
import xarray as xr

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
from sudsaq.data  import load
from sudsaq.ml    import plots
from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)

#%%

model  = load_pkl('local/runs/1.create-rf/fold_0/model.pkl')
data   = xr.open_dataarray('local/runs/1.create-rf/fold_0/test.data.nc')
target = xr.open_dataarray('local/runs/1.create-rf/fold_0/test.target.nc')

config = Config('sudsaq/configs/dev/ml/dev.yml', 'create-rf')

data = data.stack(loc=['lat', 'lon', 'time'])
data = data.transpose('loc', 'variable')
data = data.dropna('loc')

target = target.stack(loc=['lat', 'lon', 'time'])
target = target.dropna('loc')

#%%

import xarray as xr

ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2012/07.nc')

#%%

from sudsaq.data import Dataset

ds = Dataset(ds)

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

_p, _b, _c = _aq()
p, b, c = _ti()

#%%


np.isclose(p, _p).all()
np.isclose(b, _b).all()
np.isclose(c, _c).all()

#%%
 p[-1].values
_p[-1].values
p[-1].values == _p[-1].values

#%%
_p[-1]
p[-1]

{**config.treeinterpreter}

2**16
