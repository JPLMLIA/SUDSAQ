


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

ds
