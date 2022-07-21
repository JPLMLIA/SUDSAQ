

import xarray as xr


ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/201[0-4]/*.nc', parallel=True)
ds = xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/data/momo/201[0-4]/*.nc', parallel=True)
#%%

ms = ds.mean('time')
ms.visualize()

#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import dask

dask.visualize(ms['momo.o3'])

ms['momo.o3']

ds.close()


#%%

sum([45, 35, 25, 10]) * 2 + 5*4
sum([45, 35, 25, 10, 5, 2.5]) * 2
