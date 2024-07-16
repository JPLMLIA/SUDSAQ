#%%
#%%
#%%
import xarray as xr

momo = xr.open_dataset('')
python createJob.py -p "default<-gattaca<-v4<-bias-median<-toar-v2.3<-RFQ<-extended-no-france-2014"

#%%

import numpy as np
import pandas as pd
import xarray as xr

# Sample data: creating a 3D array with dimensions (time, lat, lon)
# Replace this with your actual geospatial-temporal dataset
data = np.random.rand(10, 20, 20)  # Example data, 10 time steps, 20x20 grid

# Creating xarray DataArray
times = pd.date_range('2024-01-01', periods=10)
lats = np.linspace(-90, 90, 20)
lons = np.linspace(-180, 180, 20)
da = xr.DataArray(data, dims=('time', 'lat', 'lon'), coords={'time': times, 'lat': lats, 'lon': lons})

# Define graph connections based on spatial relationships
# For simplicity, let's consider a fully connected graph
num_nodes = len(lats) * len(lons)
adjacency_matrix = np.ones((num_nodes, num_nodes))  # Fully connected graph
np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections

da

# Convert adjacency matrix to PyTorch tensor
#%%
adjacency_tensor = torch.FloatTensor(adjacency_matrix)

#%%

from mlky import Config
Config("sudsaq/configs/definitions.yml", "default<-v4<-v6r<-extended-no-france-2014")
Config.input.glob
