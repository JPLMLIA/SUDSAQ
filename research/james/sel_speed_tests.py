import xarray as xr
from timeit import timeit

ns = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/*[0-7]/*.nc', parallel=True, engine='scipy') # 31 seconds

#%%

def method_1():
    ps = ns.sel(time=slice('2013', '2014'))
    ps = ps.sel(time=ps.time.dt.month == 1)

def method_2():
    ns.sel(time=((ns.time.dt.year == 2013) | (ns.time.dt.year == 2014)) & (ns.time.dt.month == 1))

def method_3():
    ns.sel(time=ns.time.dt.year.isin([2013, 2014]) & ns.time.dt.month.isin([1]))

def method_4():
    regex = '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/*201[3-4]/07.nc'
    xr.open_mfdataset(regex, parallel=True, engine='scipy')

#%%
# Run once
timeit(method_1, number=1, globals={'ns': ns})
timeit(method_2, number=1, globals={'ns': ns})
timeit(method_3, number=1, globals={'ns': ns})
timeit(method_4, number=1, globals={'ns': ns})

#%%
# Run 100 times
timeit(method_1, number=100, globals={'ns': ns})
timeit(method_2, number=100, globals={'ns': ns})
timeit(method_3, number=100, globals={'ns': ns})
timeit(method_4, number=100, globals={'ns': ns})
