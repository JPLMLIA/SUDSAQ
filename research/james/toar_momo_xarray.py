#%%

# pip install xarray dask
# Dask is needed for lazy loading via xr.open_mfdataset

#%%
import xarray as xr

from glob import glob

path  = '/Volumes/MLIA_active_data/data_SUDSAQ/data'
momo  = glob(f'{path}/momo/**/*.nc')                    # List of MOMO .nc files
toar  = glob(f'{path}/toar/matched/**/*.nc')            # List of TOAR .nc files
files = momo + toar
len(files)

ds = xr.open_mfdataset(files, engine='scipy', parallel=True) # Lazy load in parallel
ds

#%%

ns = ds.sel(time='2012-06') # Select only June of 2012
ns = ns[['co', 'toar/o3/dma8epa/mean', 'toar/o3/dma8epa/std', 'toar/o3/dma8epa/count']] # Select only these variables
ns

#%%

ns.load() # Now load the data into memory
ns

#%%
# The times of TOAR do not align with the times of MOMO
(~ns.dropna('time', subset=['toar/o3/dma8epa/count']).isnull()).any()

#%%
# Resample
ts = ns.groupby('time.day').mean()
ts

#%%

stacked = ts.stack({'stacked': ['lat', 'lon']})
stacked

#%%

da = ts.to_array()
da.values

#%%

da.stack({'data': ['day', 'variable']})
da.stack({'data': ['day', 'variable'], 'loc': ['lat', 'lon']})

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
# Simple plotting process
ax = plt.subplot(projection=ccrs.PlateCarree())
ts['co'].mean('day').plot(ax=ax) # Data must be 2d, so take the mean of the `days` dimension
ax.coastlines()
ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)

#%%

geospatial(ts['co'].mean('day'))
geospatial(ts['toar/o3/dma8epa/mean'].mean('day'))
geospatial(ts['toar/o3/dma8epa/std'].mean('day'))
geospatial(ts['toar/o3/dma8epa/count'].mean('day'))

#%%
# Convert to numpy
ds.to_array().values

type(ts['toar/o3/dma8epa/mean'].values)

#%% [Markdown]
## # Subselect MOMO data between 8am and 4pm and take the average to align with daily TOAR data

#%%
import xarray
import datetime as dt

# Load in the data
ds = xr.open_mfdataset(files, engine='scipy', parallel=True)
ns = ds.sel(time='2012-06')
ns = ns[['co', 'toar/o3/dma8epa/mean', 'toar/o3/dma8epa/std', 'toar/o3/dma8epa/count']]
ns.load()

#%%

time = ns.time.dt.time # Convert the times dimension to datetime.time objects
mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0))
# Select timestamps between 8am and 4pm OR midnight

ss = ns.where(mask, drop=True)     # Drop timestamps that don't match
gs = ss.groupby('time.day').mean() # Take the mean by day
gs
