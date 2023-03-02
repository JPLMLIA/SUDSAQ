
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
        select[dim] = sel = slice(*sel)
    else:
        Logger.debug(f'Selecting: {dim}=={sel}')
        select[dim] = sel
    Logger.debug(f'Selecting on dimension `{dim}` using: {sel}')

if select:
    ds = ds.sel(**select)

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
