#%%
#%%
#%%
import xarray as xr

mask = '/Users/jamesmo/projects/sudsaq/dev/.local/data/bias/out_mask.nc'

mask = xr.open_dataset(mask)

#%%
xr.where(mask.mask == 34.)

mask.mask[mask.mask == 23.]
mask = mask.mask
mask == 23

#%%

mask = mask['mask']
mask = mask.stack(ll=['lat', 'lon'])
#%%
for pair in mask[mask == 23.].ll:
    print(f'- {(pair.data).tolist()}')

"""
drop_coords:
(42.056, 9.0)
(43.177, 0.0)
(43.177, 1.125)
(43.177, 2.25)
(43.177, 358.875)
(44.299, 0.0)
(44.299, 1.125)
(44.299, 2.25)
(44.299, 3.375)
(44.299, 4.5)
(44.299, 5.625)
(44.299, 6.75)
(44.299, 358.875)
(45.42, 0.0)
(45.42, 1.125)
(45.42, 2.25)
(45.42, 3.375)
(45.42, 4.5)
(45.42, 5.625)
(45.42, 6.75)
(45.42, 358.875)
(46.542, 0.0)
(46.542, 1.125)
(46.542, 2.25)
(46.542, 3.375)
(46.542, 4.5)
(46.542, 5.625)
(46.542, 358.875)
(47.663, 0.0)
(47.663, 1.125)
(47.663, 2.25)
(47.663, 3.375)
(47.663, 4.5)
(47.663, 5.625)
(47.663, 6.75)
(47.663, 357.75)
(47.663, 358.875)
(48.785, 0.0)
(48.785, 1.125)
(48.785, 2.25)
(48.785, 3.375)
(48.785, 4.5)
(48.785, 5.625)
(48.785, 6.75)
(48.785, 7.875)
(48.785, 356.625)
(48.785, 358.875)
(49.906, 1.125)
(49.906, 2.25)
(49.906, 3.375)
(49.906, 4.5)
(51.028, 2.25)
"""

mask

coords = mask[mask == 23.].ll.data.tolist()

set(mask['ll'].data)

all = set(mask.ll.data.tolist())
sub = all - set(coords)
len(sub)
len(all)

mask.sel(ll=list(sub))
#%%

def sel_by_latlon_pair(ds: xr.Dataset, pairs: list, remove: bool=False) -> xr.Dataset:
    """
    Selects data from an xarray Dataset based on latitude and longitude pairs.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data.

    pairs : list
        A list of latitude and longitude pairs used for selection.

    remove : bool, optional
        If True, remove the specified pairs instead of selecting them.
        Default is False.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the selected data based on the specified pairs.

    Notes
    -----
    This function selects data from the input xarray Dataset based on latitude and longitude pairs.
    If `remove` is True, it removes the specified pairs from the selection instead of selecting them.
    """
    if remove:
        pairs = list(set(ds['loc'].data) - set(pairs))

    return ds.sel(loc=pairs)

#%%

"""
drop_coords:
coords = [
    (42.056, 9.0),
    (43.177, 0.0),
    (43.177, 1.125),
    (43.177, 2.25),
    (43.177, 358.875),
    (44.299, 0.0),
    (44.299, 1.125),
    (44.299, 2.25),
    (44.299, 3.375),
    (44.299, 4.5),
    (44.299, 5.625),
    (44.299, 6.75),
    (44.299, 358.875),
    (45.42, 0.0),
    (45.42, 1.125),
    (45.42, 2.25),
    (45.42, 3.375),
    (45.42, 4.5),
    (45.42, 5.625),
    (45.42, 6.75),
    (45.42, 358.875),
    (46.542, 0.0),
    (46.542, 1.125),
    (46.542, 2.25),
    (46.542, 3.375),
    (46.542, 4.5),
    (46.542, 5.625),
    (46.542, 358.875),
    (47.663, 0.0),
    (47.663, 1.125),
    (47.663, 2.25),
    (47.663, 3.375),
    (47.663, 4.5),
    (47.663, 5.625),
    (47.663, 6.75),
    (47.663, 357.75),
    (47.663, 358.875),
    (48.785, 0.0),
    (48.785, 1.125),
    (48.785, 2.25),
    (48.785, 3.375),
    (48.785, 4.5),
    (48.785, 5.625),
    (48.785, 6.75),
    (48.785, 7.875),
    (48.785, 356.625),
    (48.785, 358.875),
    (49.906, 1.125),
    (49.906, 2.25),
    (49.906, 3.375),
    (49.906, 4.5),
    (51.028, 2.25),
]
"""

#%%

ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/2014/without-france

#%%
from wcmatch import glob
len(list(range(2005, 2021)))
glob.glob('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/201[3-5]/?(without-france/)*.nc', flags=glob.BRACE | glob.EXTMATCH)

"20{0[5-9],1[0-9],20}"
glob.glob('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/{201{3,5},2014/without-france}/*.nc', flags=glob.BRACE)
glob.glob('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/{20{0[5-9],1{[1-3],[5-9]},20},2014/without-france}/*.nc', flags=glob.BRACE)

#%%

ls /Volumes/MLIA_active_data/data_SUDSAQ/models/bias/

#%%
import logging
logging.basicConfig(level=logging.DEBUG)
from mlky import Config

Config("sudsaq/configs/definitions.yml", "default<-mac<-bias-median<-v4<-v6r<-toar-v2.3<-dev-limited<-jan", debug=[0])

#%%
Config.target

from mlky import Null
Config.get('target', var=True).replace(Null)

Config.get('target', var=True).reset()
from mlky.magics import replace

import mlky

dir(mlky)

from mlky import magics
magics.replace(Config.target)

#%%
import xarray as xr

ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/2014/01.nc').load()

mask = '/Users/jamesmo/projects/sudsaq/dev/.local/data/bias/out_mask.nc'
mask = xr.open_dataset(mask).mask

#%%
mask = ~(mask == 23)



#%%
import numpy as np

ns = ds.where(mask, np.nan, ds)

ns = xr.where(mask == 23, np.nan, ds)
xr.where(mask == 23, np.nan, mask)
ds.where(mask == 23)
ns.to_netcdf('without-france.nc')

#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def geospatial(data, title=None):
    """
    """
    # Prepare figure
    fig = plt.figure(figsize=(15, 8))
    ax  = plt.subplot(projection=ccrs.PlateCarree())

    # data.plot.pcolormesh(x='lon', y='lat', ax=ax, levels=10, cmap='viridis')
    data.plot(ax=ax)

    ax.coastlines()
    ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)

    if title:
        ax.set_title(title)
    plt.show()

#%%
geospatial(
    ds['toar.mda8.median'].isel(time=0)
)

#%%
geospatial(
    ds.where(mask != 23)['toar.mda8.median'].isel(time=0)
)

#%%





import xarray as xr

ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2.3/2014/*.nc')

mask = '/Users/jamesmo/projects/sudsaq/dev/.local/data/bias/out_mask.nc'
mask = xr.open_dataset(mask).mask

lon = mask.lon.values
lon[lon > 180] -= 360
mask['lon'] = lon

ns = ds.where(mask != 23)
ns

#%%

from sudsaq.data import save_by_month

mkdir without-france
save_by_month(ns, 'without-france')
