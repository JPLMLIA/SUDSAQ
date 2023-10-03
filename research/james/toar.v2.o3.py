#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TOAR v2 querying script dev
import json
import pandas as pd
import requests

from io   import StringIO
from mlky import Section
from tqdm import tqdm

from sudsaq.utils import save_pkl

def query(url, api, flags={}, pre='', timeout=60, verbose=True):
    """
    """
    params = ''
    if flags:
        params = '?' + ('&'.join([f'{k}={v}' for k, v in flags.items()])).replace(':', '%3A').replace('+', '%2B')
    request = f'{url}/{api}/{pre}{params}'

    if verbose:
        print(f'request({request!r})')

    try:
        r = requests.get(request, timeout=timeout)
    except:
        return
    if not r:
        if verbose:
            print(f'Request failed, reason: {r.reason}')
    return r


# Base URL for TOAR API v2
url = 'https://toar-data.fz-juelich.de/api/v2'

# Retrieve the o3 ID
req = query(url, 'variables', pre='o3')

#%%
o3  = req.json()['id']

# Retrieve the list of stations
stations = Section(
    name = 'Stations',
    data = query(url, 'stationmeta', {
        'limit' : None,
        'fields': ','.join([
            'id',
            'name',
            'coordinates'
        ])
    }).json()
)

# Retrieve all the timeseries records for this variable
ts = query(url, 'search', {
    'limit': None,
    'variable_id': o3,
    'fields': ','.join([
        'id',
        'data_start_date',
        'data_end_date',
    ])
}).json()

# Parse date to Pandas for easier comparisons
start, end = '2005-01-01', '2020-12-31'
start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')

#%%
# Subselect only timeseries that fall in range
ids = []
for record in ts:
    dstart = pd.Timestamp(record['data_start_date'])
    dend   = pd.Timestamp(record['data_end_date'])
    if any([
        ( start <= dstart) & (dstart <=  end), # Starts in range
        ( start <= dend)   & (  dend <=  end), # Ends in range
        (dstart <=  start) & (   end <= dend)  # Spans range
    ]):
        ids.append(record['id'])

print(f'Found {len(ids)} of {len(ts)} total records - {len(ids)/len(ts)*100:.2f}%')
ids.sort()

#%%

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def mda(ts, between=(8, 16), inclusive=True):
    """
    """
    hours = ts.time.dt.hour
    return ts[(between[0] <= hours) & (hours <= between[1])].mean()

print(f'{len(ids)} ids to process')
failed = []
for id in tqdm(ids, desc='Downloading TOAR2 data'):
    req = query(url, f'data/timeseries', pre=id, verbose=False, timeout=5, flags={
        'format': 'csv'
        }
    )
    if req:
        data = req.text

        try:
            df = pd.read_csv(StringIO(data),
                comment     = '#',
                header      = 1,
                sep         = ',',
                names       = ["time", "value", "flags", "version", "timeseries_id"],
                parse_dates = ["time"],
                index_col   = "time",
                date_parser = lambda date: pd.to_datetime(date, utc=True), infer_datetime_format=True
            ).drop(columns=['flags', 'version'])

            # Extract the metadata
            meta = json.loads("\n".join([line[1:] for line in data.split('\n') if line.startswith('#')]))
            df['lat'] = meta['station']['coordinates']['lat']
            df['lon'] = meta['station']['coordinates']['lng']

            # Calculate maximum daily average over 8 hours - mda8
            df = df.groupby(df.time.dt.floor('d')).apply(mda)

            df.to_hdf('o3.mda8.2017.h5', f'TS_ID_{id}')
            del df
        except:
            failed.append(id)
    else:
        failed.append(id)

print(f'Finished. {len(failed)} failed and need to be reprocessed.')
save_pkl(failed, 'failed_ids.pkl')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Combine toar tracks dataframes to one
import h5py
import pandas as pd

from tqdm import tqdm

file = '.local/data/toar/v2.mda8.h5'

with h5py.File(file, 'r') as h5:
    keys = list(h5.keys())

dfs = []
for key in tqdm(keys):
    df = pd.read_hdf(file, key)
    df = df.reset_index()
    df = df.loc[df['time'] >= "2005"]
    dfs.append(df)

df = pd.concat(dfs)
df

#% Drop NaNs
df = df.dropna(subset=['value'])
df

#% Sort
df = df.sort_values('timeseries_id')
df

#% Drop duplicates
df = df.drop_duplicates(subset=['time', 'lat', 'lon'], keep='last')
df.sort_values('time')

#% Rename
df = df.rename(columns={'value': 'o3.mda8'})
df = pd.read_hdf('.local/data/toar/mda8.h5', 'mda8')

#%% Save
df.to_hdf('.local/data/toar/mda8.h5', 'mda8')

# This h5 can now be fed to silos/toar/matched.py, see definitions.yml for an example

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Analysis
# Loading
import xarray as xr

# Output matched.py as a single file by disabling save_by_month
ds = xr.load_dataset('.local/data/toar/v2.matched.nc')

#%% Sometimes the timestamps don't get converted properly in the match script
ds['time'] = pd.DatetimeIndex(ds.time)

#%% Functions work best on the DataArray instead of the Dataset
da = ds['toar.mda8.mean']

#%% General plotting function, useful in notebook environments like this

def draw(data, ax=None, figsize=(13, 7), title=None, coastlines=True, gridlines=True, **kwargs):
    """
    Portable geospatial plotting function
    """
    if ax is None:
        if 'plt' not in globals():
            global plt
            import matplotlib.pyplot as plt
        if 'ccrs' not in globals():
            global ccrs
            import cartopy.crs as ccrs

        fig = plt.figure(figsize=figsize)
        ax  = plt.subplot(111, projection=ccrs.PlateCarree())

    plot = data.plot.pcolormesh(x='lon', y='lat', ax=ax, **kwargs)

    if title:
        ax.set_title(title)
    if coastlines:
        ax.coastlines()
    if gridlines:
        ax.gridlines(draw_labels=False, color='dimgray', linewidth=0.5)

    return ax

#%% Filter extreme values
import numpy as np

filt = da.where(da > -10, np.nan)

draw(filt.mean('time'))


#%% Load v1 TOAR for comparing against

v2 = filt # Easy reference name
v1 = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2*/*.nc')
v1 = v1['toar.o3.dma8epa.mean'].load().sortby('lon')
v1.name = 'v1'
v2.name = 'v2' # For plotting

#%% Sample counts per station

_ = draw(v1.count('time'), title=f"Station Count = {int(v1.mean('time').count())}")
_ = draw(v2.count('time'), title=f"Station Count = {int(v2.mean('time').count())}")


#%% GIF side-by-side of station counts by year

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def update(i):
    """
    """
    year = None

    ax1.clear()
    if i < y1c.year.size:
        year = int(y1c.year[i])
        data = y1c.isel(year=i)
        draw(
            data,
            ax    = ax1,
            add_colorbar = False,
            vmax  = vmax,
            vmin  = vmin,
            title = f'v1 - Total Count = {int(data.sum())}'
        )
    else:
        draw(empty, ax=ax1, add_colorbar=False, vmin=vmin, vmax=vmax, title=f'v1 - Total Count = 0')

    ax2.clear()
    if i < y2c.year.size:
        year = int(y2c.year[i])
        data = y2c.isel(year=i)
        draw(
            data,
            ax    = ax2,
            add_colorbar = False,
            vmax  = vmax,
            vmin  = vmin,
            title = f'v2 - Total Count = {int(data.sum())}'
        )
    else:
        draw(empty, ax=ax2, add_colorbar=False, vmin=vmin, vmax=vmax, title=f'v2 - Total Count = 0')

    fig.suptitle(f'Year: {year}', fontsize=42)
    print(f'Processed {year}')

    # Reduces width between plots for a cleaner look
    plt.subplots_adjust(wspace=.02, hspace=0)


plt.close('all')

y1c = v1.groupby('time.year').count()
y2c = v2.groupby('time.year').count()
empty = xr.zeros_like(y1c).isel(year=0)
vmax = int(max(y1c.max(), y2c.max()))
vmin = int(min(y1c.min(), y2c.min()))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})

frames = max(y1c.year.size, y2c.year.size)
ani = FuncAnimation(fig, update, frames=frames)

ani.save('toar.counts_by_year.gif', dpi=300, writer=PillowWriter(fps=1))

#%% gain/loss gif
from matplotlib.animation import FuncAnimation, PillowWriter

def update(i):
    """
    """
    # Retrieve how many values should be in this year
    year = str(int(c2.isel(year=i).year))
    vmax = max(v1.sel(time=year).time.size, v2.sel(time=year).time.size)
    vmin = -vmax

    # Plot limits
    # Levels corrospond to [Completely new, some new, some lost, completely lost]
    levels = [vmin, vmin+1, -1, 1, vmax-1, vmax]

    ax.clear()

    diff = c1.isel(year=i) - c2.isel(year=i)
    draw(
        diff,
        ax = ax,
        add_colorbar = False,
        vmax  = vmax,
        vmin  = vmin,
        cmap  = 'bwr',
        title = int(year),
        levels = levels
    )

    plt.tight_layout()

plt.close('all')

# Counts
c1 = v1.groupby('time.year').count()
c2 = v2.groupby('time.year').count()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})

frames = min(c1.year.size, c2.year.size)
ani = FuncAnimation(fig, update, frames=frames)

ani.save('toar.differences.gif', dpi=300, writer=PillowWriter(fps=.3))


#%%
from sudsaq.data import save_by_month

save_by_month(ds, '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2/')

#%% Histograms

_ = v1.plot.hist(figsize=(13, 5), yscale='log')
_ = v2.plot.hist(figsize=(13, 5), yscale='log')

#%%



#%%

v1.min(), v1.max(), v1.mean(), v1.std()
v2.min(), v2.max(), v2.mean(), v2.std() # v2 == filtered
bounded.min(), bounded.max(), bounded.mean(), bounded.std()

#%%
v1c = v1.count()
v2c = v2.count()
fixed = bounded.count()

(1 - count / v2c) * 100

count / v1c * 100

#%%

concern = v2.where(v2 < -10, np.nan)
draw(concern.count('time'), title=f'Count of Values less than -10, total = {int(concern.count())}', levels=[-2, 0, .1], cmap='bwr')
#%%
(10665 - 9)/10665
out_bounds = v2.where(
    (v1.min() > v2) | (v2 > v1.max()),
    np.nan
)
stations = int(out_bounds.mean('time').count())
draw(out_bounds.count('time'), title=f'Stations with a value out of v1 bounds, total = {stations}', levels=[-2, 0, .1], cmap='bwr')

#%%

fixed = v2.where(v2 > -10, np.nan)
draw(fixed.count('time'), title=f'Count of Values less than -10, total = {int(fixed.count())}')
fixed.plot.hist(figsize=(13, 5), yscale='log')

#%%

bounded = v2.where(
    (v1.min() <= v2) & (v2 <= v1.max()),
    np.nan
)
bounded.plot.hist(figsize=(13, 5), yscale='log')


#%%


save_by_month(ds, '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2/')

v22.to_netcdf('.local/data/toar/fixed/fixed.nc')
bounded

ds
fixed
v22['toar.mda8.mean'][:] = fixed.values

v22

#%%

- Integrate new TOAR data using v1 bounded
- v4<-bias[median]

#%%
import numpy  as np
import xarray as xr


ds = xr.load_dataset('.local/data/toar/v2.matched.nc')

filt = ds.where(ds > -10, np.nan)

#%%
from sudsaq.data import save_by_month

save_by_month(filt, '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/v2/')

#%%


f = ['t', 'a/b']



{key: 'momo.'+key.replace('/', '.') for key in f}
