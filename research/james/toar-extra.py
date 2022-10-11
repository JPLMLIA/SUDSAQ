#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import xarray as xr

#%%
ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar
ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc')

#%%

ds

md = xr.open_dataset('local/data/dev/toar/metadata.test.nc')
md

#%%

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc',
    '/Users/jamesmo/projects/suds-air-quality/local/data/dev/toar/metadata.test.nc'
]

ds = xr.open_mfdataset(files, parallel=True)

ds.load()

#%%
data = ds.stack({'loc': ['lat', 'lon', 'time']})
data

#%%
data['toar.station_alt.std'].mean()
md['toar.station_alt.std'].mean()

data.isel(loc=50600)

md.isel(lat=50, lon=50)

md['toar.station_alt.mean'].where(~md['toar.station_alt.mean'].isnull())

#%%

ds.sel(lon=[])

#%%

tl = ds.drop_dims('time')

#%%

-140+180, -50+180

convert = lambda lon1, lon2: (lon1 + 360 if lon1 < 0 else lon1, lon2 + 360 if lon2 < 0 else lon2)
convert(-140, -50)
convert(-5, 5)

220 < lat < 310
355 < lat < 5

ds

#%%

ns = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/01.nc')
ns
#%%

ns.sel(lon=5)
ns.where(ns.lon<5, ns.lon>355, drop=True)

ns.sel({'lon': slice(355, 360), 'lon': slice(0, 5)})

#%%
# ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/


#%%

data

ds
ns


sum([d.lon.size for d in data])

#%%

s = set(ds.lon.values)


s - set(ns.lon.values)

#%%


sub.time +

sub.time




help(np.timedelta64)


#%%

ns.drop_dims('time').any()

ps = ns.drop_dims('time')

len(ns)
len(ps)
ns

list(ns)
len(ps)
ps


a = ns.to_array()

a['variable']


#%%

for v in a['variable']:
    d = a.sel(variable=v)

d

#%%

>>> path  = 'Z:/data_SUDSAQ/model/new/model_data\\*\\combined\\test.data.nc'
>>> split = path.split('\\')
>>> split
['Z:/data_SUDSAQ/model/new/model_data', '*', 'combined', 'test.data.nc']
>>> split[1]
'*'

"""
2022-09-02 12:53:00,741 WARNING worker.py:1829 -- The node with node id: 5ad7fa5dd64fc1fec7a82cceecabed8e5eb247ec97064bb50a1c2a11 and address: 137.78.248.21 and node name: 137.78.248.21 has been marked dead because the detector has missed too many heartbeats from it. This can happen when a    (1) raylet crashes unexpectedly (OOM, preempted node, etc.)
        (2) raylet has lagging heartbeats due to slow network or busy workload.
./monthly.sh: line 32: 3376060 Killed                  nice -n 19 python ml/create.py -c $config -s $section
"""

#%%

str(ds['variable'])

list(ds)
