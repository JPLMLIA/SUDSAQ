#%%
#%%
#%%
# xarray documentation:
# Import xarray, if not installed run `pip install xarray` in your Python (Conda) environment
import xarray as xr

# Load one MOMO file
# Change /Volumes/ to /data/ if on an MLIA machine
# We normally use the variables `ds`, `ns` because the object is a Dataset (ds) or a New Dataset (ns)
ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/01.nc')

print(ds) # This will produce the pasted below
"""
<xarray.Dataset>
Dimensions:                  (time: 372, lat: 160, lon: 320)
Coordinates:
  * time                     (time) datetime64[ns] 2005-01-01T01:00:00 ... 20...
  * lon                      (lon) float64 0.0 1.125 2.25 ... 356.6 357.8 358.9
  * lat                      (lat) float64 -89.14 -88.03 -86.91 ... 88.03 89.14
Data variables: (12/145)
    momo.2dsfc.Br2           (time, lat, lon) float32 ...
    momo.2dsfc.BrCl          (time, lat, lon) float32 ...
    momo.2dsfc.BrONO2        (time, lat, lon) float32 ...
    momo.2dsfc.BrOX          (time, lat, lon) float32 ...
    momo.2dsfc.C10H16        (time, lat, lon) float32 ...
    momo.2dsfc.C2H5OOH       (time, lat, lon) float32 ...
    ...                       ...
    momo.so2                 (time, lat, lon) float32 ...
    momo.t                   (time, lat, lon) float32 ...
    momo.u                   (time, lat, lon) float32 ...
    momo.v                   (time, lat, lon) float32 ...
    momo.mda8                (time, lat, lon) float32 ...
    momo.o3                  (time, lat, lon) float32 ...
"""
#%%
# The shape of the object is 3d (372, 160, 320)
# There are 145 variables total
# The file is "lazy loaded" which means the keys to the file are loaded but the data itself is not
# This is because this file alone is 10gb and would be difficult to load into memory
# We can subselect which variables to work with by passing in a list
ns = ds[['momo.mda8', 'momo.o3']]
print(ns)
"""
<xarray.Dataset>
Dimensions:    (time: 372, lat: 160, lon: 320)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-01T01:00:00 ... 2005-01-31T23:00:00
  * lon        (lon) float64 0.0 1.125 2.25 3.375 ... 355.5 356.6 357.8 358.9
  * lat        (lat) float64 -89.14 -88.03 -86.91 -85.79 ... 86.91 88.03 89.14
Data variables:
    momo.mda8  (time, lat, lon) float32 ...
    momo.o3    (time, lat, lon) float32 ...
"""
#%%
# Now that it's much smaller, we can load the file
ns.load()
print(ns)
"""
<xarray.Dataset>
Dimensions:    (time: 372, lat: 160, lon: 320)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-01T01:00:00 ... 2005-01-31T23:00:00
  * lon        (lon) float64 0.0 1.125 2.25 3.375 ... 355.5 356.6 357.8 358.9
  * lat        (lat) float64 -89.14 -88.03 -86.91 -85.79 ... 86.91 88.03 89.14
Data variables:
    momo.mda8  (time, lat, lon) float32 17.64 17.65 17.65 17.65 ... nan nan nan
    momo.o3    (time, lat, lon) float32 16.81 16.82 16.83 ... 21.24 21.2 21.16
"""
# Now the variables are loaded so we can apply operations
#%%
## Example on some basic subselecting operations (https://xarray.pydata.org/en/stable/user-guide/indexing.html)
# Select only one date
ns.sel(time='2005-01-01')
"""
<xarray.Dataset>
Dimensions:    (time: 12, lat: 160, lon: 320)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-01T01:00:00 ... 2005-01-01T23:00:00
  * lon        (lon) float64 0.0 1.125 2.25 3.375 ... 355.5 356.6 357.8 358.9
  * lat        (lat) float64 -89.14 -88.03 -86.91 -85.79 ... 86.91 88.03 89.14
Data variables:
    momo.mda8  (time, lat, lon) float32 17.64 17.65 17.65 17.65 ... nan nan nan
    momo.o3    (time, lat, lon) float32 16.81 16.82 16.83 ... 16.04 16.06 16.09
"""
# One day is 12 timestamps because MOMO data is bihourly

# Select multiple dates
ns.sel(time=slice('2005-01-10', '2005-01-20'))
"""
<xarray.Dataset>
Dimensions:    (time: 132, lat: 160, lon: 320)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-10T01:00:00 ... 2005-01-20T23:00:00
  * lon        (lon) float64 0.0 1.125 2.25 3.375 ... 355.5 356.6 357.8 358.9
  * lat        (lat) float64 -89.14 -88.03 -86.91 -85.79 ... 86.91 88.03 89.14
Data variables:
    momo.mda8  (time, lat, lon) float32 22.21 22.23 22.25 22.27 ... nan nan nan
    momo.o3    (time, lat, lon) float32 17.84 17.87 17.88 ... 19.91 19.93 19.95
"""

# Select only North America
ns.sel(lat=slice(10, 80), lon=slice(220, 310))
"""
<xarray.Dataset>
Dimensions:    (time: 372, lat: 62, lon: 80)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-01T01:00:00 ... 2005-01-31T23:00:00
  * lon        (lon) float64 220.5 221.6 222.8 223.9 ... 306.0 307.1 308.2 309.4
  * lat        (lat) float64 10.65 11.78 12.9 14.02 ... 75.7 76.82 77.94 79.06
Data variables:
    momo.mda8  (time, lat, lon) float32 24.3 25.7 27.46 27.85 ... nan nan nan
    momo.o3    (time, lat, lon) float32 19.32 19.2 18.2 ... 26.71 26.34 26.03
"""
#%%
## Basic calculation operations (https://xarray.pydata.org/en/stable/user-guide/computation.html)
# Daily average
ns.resample(time='1D')
"""
DatasetResample, grouped over '__resample_dim__'
31 groups with labels 2005-01-01, ..., 2005-01-31.
"""
ns.resample(time='1D').mean()
"""
<xarray.Dataset>
Dimensions:    (time: 31, lat: 160, lon: 320)
Coordinates:
  * time       (time) datetime64[ns] 2005-01-01 2005-01-02 ... 2005-01-31
  * lon        (lon) float64 0.0 1.125 2.25 3.375 ... 355.5 356.6 357.8 358.9
  * lat        (lat) float64 -89.14 -88.03 -86.91 -85.79 ... 86.91 88.03 89.14
Data variables:
    momo.mda8  (time, lat, lon) float32 17.64 17.65 17.65 ... 19.61 19.62 19.63
    momo.o3    (time, lat, lon) float32 18.57 18.57 18.56 ... 20.76 20.74 20.72
"""
#%%
## Now let's load multiple files, but we'll switch to TOAR as they're smaller
# Use xr.open_mfdataset which is multiple_file_dataset
# xr.open_mfdataset requires to have Dask installed via `pip install dask`
# This can take in a regex to retrieve specific files
# This will load in data for years [2010, 2011, 2012] only for July
ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/201[0-2]/07.nc', parallel=True)
print(ds)
"""
<xarray.Dataset>
Dimensions:                (lon: 320, lat: 160, time: 93)
Coordinates:
  * lon                    (lon) float64 0.0 1.125 2.25 ... 356.6 357.8 358.9
  * lat                    (lat) float64 -89.14 -88.03 -86.91 ... 88.03 89.14
  * time                   (time) datetime64[ns] 2010-07-01 ... 2012-07-31
Data variables:
    toar.o3.dma8epa.mean   (time, lat, lon) float64 dask.array<chunksize=(31, 160, 320), meta=np.ndarray>
    toar.o3.dma8epa.std    (time, lat, lon) float64 dask.array<chunksize=(31, 160, 320), meta=np.ndarray>
    toar.o3.dma8epa.count  (time, lat, lon) float64 dask.array<chunksize=(31, 160, 320), meta=np.ndarray>
"""
# Again, the files are lazy loaded, but this time using Dask
ds['toar.o3.dma8epa.mean']
"""
xarray.DataArray'toar.o3.dma8epa.mean'
time: 93 lat: 160 lon: 320
	       Array	Chunk
Bytes	36.33 MiB	12.11 MiB
Shape	(93, 160, 320)	(31, 160, 320)
Count	9 Tasks	3 Chunks
Type	float64	numpy.ndarray
"""
# The array is "chunked" so that it won't load everything in at once if an operation that requires loading were to happen
# There are 3 chunks of about 12.11MB each. Think of each input file as its own chunk, because we chose 1 month over 3 years, there are 3 files so 3 chunks
# If we were to load all the chunks in, it'd be 36.33MB of memory. This is pretty small so it's safe to do so
ds.load()
print(ds)
"""
<xarray.Dataset>
Dimensions:                (lon: 320, lat: 160, time: 93)
Coordinates:
  * lon                    (lon) float64 0.0 1.125 2.25 ... 356.6 357.8 358.9
  * lat                    (lat) float64 -89.14 -88.03 -86.91 ... 88.03 89.14
  * time                   (time) datetime64[ns] 2010-07-01 ... 2012-07-31
Data variables:
    toar.o3.dma8epa.mean   (time, lat, lon) float64 nan nan nan ... nan nan nan
    toar.o3.dma8epa.std    (time, lat, lon) float64 nan nan nan ... nan nan nan
    toar.o3.dma8epa.count  (time, lat, lon) float64 0.0 0.0 0.0 ... 0.0 0.0 0.0
"""
# Let's take the average value
ds.mean()
"""
<xarray.Dataset>
Dimensions:                ()
Data variables:
    toar.o3.dma8epa.mean   float64 40.89
    toar.o3.dma8epa.std    float64 2.23
    toar.o3.dma8epa.count  float64 0.105
"""
#%%
## The above examples should help understand how to do a 5 year average. Change the regex to cover 5 years of data to load in, do some subselections
# so that the total data being loaded in isn't too large, then call .mean() on the dataset
# Feel free to send a Slack message to James if you have any questions
