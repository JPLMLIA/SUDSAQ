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

v2.test.X.drop_vars('variable')
v2.og = {}
v2.og.predict    = xr.zeros_like(v2.test.X.isel(variable=0).drop_vars('variable'))
v2.og.predict[:] = og.model.predict(v2.test.X)

assert mean_squared_error(v2.test.y, v2.og.predict.sel(loc=v2.test.y['loc']), squared=False) == og.rmse, 'Predicting using v2 data with og model did not match RMSE to og'
