
import xarray as xr

ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2007/11.nc')
ds = ds[['momo.v', 'momo.u']]
ds

ds['lon_alt'] =

lon = ds.lon
lon[lon > 180]
lon[lon > 180] -=  360

lon[lon > 180]


sel = [340, 40]

sel[0] > sel[1]

ns = ds.sel(lon=).isel(time=0)
ns.load()

ns['momo.u'].plot()

#%%
xr.__version__
select = {}
select['lon'] = (sel[0] < ds['lon']) | (ds['lon'] < sel[1])

select['lon']

ds.sel(**select)

ds.where(ds['lon'][select['lon']])



#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%

input = {
    'lat': [25, 80],
    'lon': [340, 40]
}

#%%

for dim, sel in config.input.sel.items():
    if dim == 'vars':
        Logger.debug(f'Selecting variables: {sel}')
        ds = ds[sel]
    elif dim == 'month':
        Logger.debug(f'Selecting month=={sel}')
        ds = ds.sel(time=ds['time.month']==sel)
    elif dim in ['lat', 'lon']:
        if isinstance(sel, list) and sel[0] > sel[1]:
    else:
        if isinstance(sel, list):
            sel = slice(*sel)
        Logger.debug(f'Selecting on dimension {dim} using {sel}')
        ds = ds.sel(**{dim: sel})

#%%
select = {}
for dim, sel in input.items():
    if dim == 'vars':
        Logger.debug(f'Selecting variables: {sel}')
        ds = ds[sel]

    elif dim == 'month':
        Logger.debug(f'Selecting: month=={sel}')
        ds = ds.sel(time=ds['time.month']==sel)

    elif dim in ['lat', 'lon']:
        if isinstance(sel, list) and sel[1] < sel[0]:
            Logger.debug(f'Selecting: {sel[0]} < {dim} < {sel[1]}')
            ds = ds.where(ds[dim][(sel[0] < ds[dim]) | (ds[dim] < sel[1])])

    elif isinstance(sel, list):
        Logger.debug(f'Selecting: {dim}[{sel[0]}:{sel[1]}]')
        ds = ds.sel(**{dim, slice(*sel)})
    else:
        Logger.debug(f'Selecting: {dim}=={sel}')
        ds = ds.sel(**{dim, sel})

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%

xr.core.dataset.Dataset

type(ds)

#%%
from importlib import reload

import sudsaq

import xarray as xr


xr.core.dataset.Dataset

ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2007/11.nc')

type(ds)

#%%
point = (0, 1)
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):
        print(f"X={x}, Y={y}")
    case _:
        raise ValueError("Not a point")

#%%


dates = [
    'Day of',
    '0 days',
    '1 day',
    '7 days',
    '1 month',
    '3 months',
    'yearly'
]

yearly = 'yearly' in dates

for date in dates:
    if date.lower() == 'day of':
        date = '0 days'

    match date.split(' '):
        case n, 'day' | 'days':
            print(f'Reminder is {n} days')
        case n, 'month' | 'months':
            print(f'Reminder is {n} months')



sum([[0, 1], [2, 3]])

#%%

import xarray as xr

help(xr.open_mfdataset)

from mlky import Null

dict(Null())

data
