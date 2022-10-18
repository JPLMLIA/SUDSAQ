from sudsaq.config import Section
from sudsaq.utils  import load_pkl
from sudsaq.ml     import treeinterpreter as ti
import xarray as xr
import ray; ray.init(_plasma_directory= '/scratch/jamesmo/tmp', _temp_dir= '/scratch/jamesmo/tmp')


def load_run(base='bias/8hour/optimized/v1', month='jan', year=2012, train=False):
    run = Section('Optimized', {'test': {}, 'train': {}})
    if train:
        run.train.data    = xr.open_dataset(f'{base}/{month}/rf/{year}/train.data.nc')
        run.train.target  = xr.open_dataarray(f'{base}/{month}/rf/{year}/train.target.nc')
        run.train.X       = run.train.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
        run.train.y       = run.train.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    run.test.data     = xr.open_dataset(f'{base}/{month}/rf/{year}/test.data.nc')
    run.test.target   = xr.open_dataarray(f'{base}/{month}/rf/{year}/test.target.nc')
    run.test.predict  = xr.open_dataarray(f'{base}/{month}/rf/{year}/test.predict.nc')
    run.test.X        = run.test.data.to_array().stack(loc=['lat', 'lon', 'time']).dropna('loc').transpose('loc', 'variable')
    run.test.y        = run.test.target.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    run.test.yh       = run.test.predict.stack(loc=['lat', 'lon', 'time']).dropna('loc')
    _, run.aligned    = xr.align(run.test.y, run.test.yh)
    run.model         = load_pkl(f'{base}/{month}/rf/{year}/model.pkl')
    # Calculate true predict
    run.predict    = xr.zeros_like(run.test.X.isel(variable=0).drop_vars('variable'))
    run.predict[:] = run.model.predict(run.test.X)
    return run

run = load_run('/data/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias/8hour/optimized/v1')

def test_ti(ret=False):
    predict    = xr.zeros_like(run.test.X.isel(variable=0).drop_vars('variable'))
    predicts   = ti.predict(run.model, run.test.X, n_jobs=-1)[0]
    predict[:] = predicts.flatten()
    if ret:
        return predict
    return run.predict.identical(predict)

print(test_ti())
