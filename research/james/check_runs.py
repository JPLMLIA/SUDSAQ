# cd /data/MLIA_active_data/data_SUDSAQ/models/2011-2015/

import xarray as xr
from sudsaq.config import Section
from sudsaq.utils import load_pkl


# Load original
og = Section('Original', {'test': {}, 'train': {}})

og.test.data   = xr.open_dataset('bias-8hour/jan/rf/2012/test.data.nc')
og.test.target = xr.open_dataarray('bias-8hour/jan/rf/2012/test.target.nc')
og.test.predict = xr.open_dataarray('bias-8hour/jan/rf/2012/test.predict.nc')

og.test.X  = og.test.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
og.test.y  = og.test.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
og.test.yh = og.test.predict.stack(loc=['lat', 'lon', 'time']).dropna('loc')

og.model = load_pkl('bias-8hour/jan/rf/2012/model.pkl')

# Load v2
v2 = Section('v2', {'test': {}, 'train': {}})

v2.test.data    = xr.open_dataset('bias-8hour-v2/jan/rf/2012/test.data.nc')
v2.test.target  = xr.open_dataarray('bias-8hour-v2/jan/rf/2012/test.target.nc')
v2.test.predict = xr.open_dataarray('bias-8hour-v2/jan/rf/2012/test.predict.nc')

v2.test.X  = v2.test.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
v2.test.y  = v2.test.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
v2.test.yh = v2.test.predict.stack(loc=['lat', 'lon', 'time']).dropna('loc')

v2.model = load_pkl('bias-8hour-v2/jan/rf/2012/model.pkl')

# Some basic checks
assert og.test.data.identical(v2.test.data), 'Not identical: data'
assert og.test.target.identical(v2.test.target), 'Not identical: target'
assert og.test.X.identical(v2.test.X), 'Not identical: X'
assert og.test.y.identical(v2.test.y), 'Not identical: y'
og.test.predict.identical(v2.test.predict)
og.test.yh.identical(v2.test.yh)

# Do some RMSE checks
from sklearn.metrics import mean_squared_error
og.rmse = mean_squared_error(og.test.y, og.test.yh.sel(loc=og.test.y['loc']), squared=False)
v2.rmse = mean_squared_error(v2.test.y, v2.test.yh.sel(loc=v2.test.y['loc']), squared=False)

v2.og = {}
v2.og.predict    = xr.zeros_like(v2.test.X.isel(variable=0).drop_vars('variable'))
v2.og.predict[:] = og.model.predict(v2.test.X)

assert mean_squared_error(v2.test.y, v2.og.predict.sel(loc=v2.test.y['loc']), squared=False) == og.rmse, 'Predicting using v2 data with og model did not match RMSE to og'

#%%

def load_v2():
    v2 = Section('v2', {'test': {}, 'train': {}})
    v2.train.data    = xr.open_dataset('bias-8hour-v2/jan/rf/2012/train.data.nc')
    v2.train.target  = xr.open_dataarray('bias-8hour-v2/jan/rf/2012/train.target.nc')
    v2.train.X       = v2.train.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
    v2.train.y       = v2.train.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    v2.test.data     = xr.open_dataset('bias-8hour-v2/jan/rf/2012/test.data.nc')
    v2.test.target   = xr.open_dataarray('bias-8hour-v2/jan/rf/2012/test.target.nc')
    v2.test.predict  = xr.open_dataarray('bias-8hour-v2/jan/rf/2012/test.predict.nc')
    v2.test.X        = v2.test.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
    v2.test.y        = v2.test.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    v2.test.yh       = v2.test.predict.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    v2.model         = load_pkl('bias-8hour-v2/jan/rf/2012/model.pkl')
    return v2

p = xr.zeros_like(v2.test.X.isel(variable=0).drop_vars('variable'))

#%%


#%%

from sudsaq.config import Section
from sudsaq.utils import load_pkl
from sklearn.ensemble import RandomForestRegressor as rf
from sudsaq.ml import treeinterpreter as ti
from sklearn.metrics import r2_score
import xarray as xr
import ray; ray.init(_plasma_directory= '/scratch/jamesmo/tmp', _temp_dir= '/scratch/jamesmo/tmp')

def load_op():
    op = Section('Optimized', {'test': {}, 'train': {}})
    op.train.data    = xr.open_dataset('bias/8hour/optimized/v1/jan/rf/2012/train.data.nc')
    op.train.target  = xr.open_dataarray('bias/8hour/optimized/v1/jan/rf/2012/train.target.nc')
    op.train.X       = op.train.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
    op.train.y       = op.train.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    op.test.data     = xr.open_dataset('bias/8hour/optimized/v1/jan/rf/2012/test.data.nc')
    op.test.target   = xr.open_dataarray('bias/8hour/optimized/v1/jan/rf/2012/test.target.nc')
    op.test.predict  = xr.open_dataarray('bias/8hour/optimized/v1/jan/rf/2012/test.predict.nc')
    op.test.X        = op.test.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
    op.test.y        = op.test.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    op.test.yh       = op.test.predict.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    _, op.aligned    = xr.align(op.test.y, op.test.yh)
    op.model         = load_pkl('bias/8hour/optimized/v1/jan/rf/2012/model.pkl')
    return op

op = load_op()

#%%

op.model
new = Section('new', {})
new.model = rf(max_features=0.33, n_estimators=20, n_jobs=-1, random_state=6789)
new.model.fit(op.train.X, op.train.y)
new.predict = xr.zeros_like(op.test.X.isel(variable=0).drop_vars('variable'))
new.predict[:] = new.model.predict(op.test.X)
new.target, new.aligned = xr.align(op.test.y, new.predict)
r2_score(new.target, new.aligned)
r2_score(new.target, op.aligned)

def ti_p(model, data):
    predict = xr.zeros_like(data.isel(variable=0).drop_vars('variable'))
    predicts, _, _ = ti.predict(model, data, n_jobs=-1)
    predict[:] = predicts.flatten()
    return predict

new.ti_p = ti_p(new.model, op.test.X)
_, new.ti_a = xr.align(op.test.y, new.ti_p)
new.predict.identical(new.ti_p)
r2_score(new.target, new.ti_a)

#%%

# op = bias/8hour/optimized/v1/jan/rf/2012
>>> op.model
RandomForestRegressor(max_features=0.33, n_estimators=20, n_jobs=-1,
                      random_state=6789)
>>> new.model = rf(max_features=0.33, n_estimators=20, n_jobs=-1, random_state=6789)
>>> new.model
RandomForestRegressor(max_features=0.33, n_estimators=20, n_jobs=-1,
                      random_state=6789)
>>> new.model.fit(op.train.X, op.train.y)
RandomForestRegressor(max_features=0.33, n_estimators=20, n_jobs=-1,
                      random_state=6789)
>>> new.predict = xr.zeros_like(op.test.X.isel(variable=0).drop_vars('variable'))
>>> new.predict[:] = new.model.predict(op.test.X)
>>> new.target, new.aligned = xr.align(op.test.y, new.predict)
>>> r2_score(new.target, new.aligned)
0.3807676532167955
>>> r2_score(new.target, op.aligned)
-0.06336675796691105
>>> def ti_p(model, data):
...     predict = xr.zeros_like(data.isel(variable=0).drop_vars('variable'))
...     predicts, _, _ = ti.predict(model, data, n_jobs=-1)
...     predict[:] = predicts.flatten()
...     return predict
...
>>> new.ti_p = ti_p(new.model, op.test.X)
>>> _, new.ti_a = xr.align(op.test.y, new.ti_p)
>>> new.predict.identical(new.ti_p)
False
>>> r2_score(new.target, new.ti_a)
0.27438791570464016
>>> new.ti_p2 = ti_p(new.model, op.test.X)
>>> new.ti_p2.identical(new.ti_p)
>>> r2_score(new.target, new.ti_a2)
0.22871956531241733

#%%
import argparse
import os

from glob import glob

from sudsaq.analyze import analyze
from sudsaq.config  import Config
from sudsaq.utils   import load_pkl

def correct(dir):
    rm = glob(f'{dir}/**/rf/**/*.bias.nc') + glob(f'{dir}/**/rf/**/*.contributions.nc')
    for file in rm:
        os.remove(file)

    [yaml] = glob(f'{dir}/*.yml')

    paths = glob(f'{dir}/**/rf/*/')
    for path in paths:
        month, _, year, _ = path.split('/')
        config = Config(yaml, month)
        config.not_ti               = True
        config.output.data          = False
        config.output.target        = False
        config.output.bias          = False
        config.output.contributions = False
        config.output.model         = False
        data   = xr.open_dataset(f'{path}/test.data.nc').to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
        target = xr.open_dataarray(f'{path}/test.target.nc').stack(loc=['lat', 'lon', 'time']).dropna('loc')
        model  = load_pkl(f'{path}/model.pkl')
        analyze(model=model, data=data, target=target, kind='test', output=path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-r', '--run',      type     = str,
                                            required = True,
                                            help     = 'Run directory to correct'
    )
    args = parser.parse_args()
    correct(args.run)
