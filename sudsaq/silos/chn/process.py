#%%
import xarray as xr

ls /Volumes/MLIA_active_data/data_SUDSAQ/CHN/*clear.nc

#%%

ds = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/CHN/*_clear.nc', parallel=True)

ds
