#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline

import psyplot.project as psy

# don't close the figures after showing them, because than the update
# would not work
# %config InlineBackend.close_figures = False
# show the figures after they are drawn or updated. This is useful
# for the visualization in the jupyter notebook
psy.rcParams['auto_show'] = True

psy.rcParams['decoder.x'] = {'lon'}
psy.rcParams['decoder.y'] = {'lat'}

#%
import xarray as xr
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')

#%%
ds = xr.open_dataset('local/runs/11-14/bias/nov/rf/2012/test.data.nc')

ds.load()
ds.close()

#%%

p = psy.plot.mapplot(ds.isel(time=0))

#%%
p = psy.plot.mapplot('local/runs/11-14/bias/nov/rf/2012/test.data.nc', name='momo.t')
psy.plot.mapplot.docs('plot')
help(psy.plot.mapplot)


ds.isel(time=0)

ds.psy.plot.mapplot(name='momo.t', cmap='Reds')


ds.dropna('lat')

import pandas as pd
type(pd.NaT)

ds.time
#%%
de = xr.open_dataset('/Users/jamesmo/Downloads/demo.nc')
de.load()
de.close()
#%%
ds.psy.plot.mapplot(name='momo.u')
de.psy.plot.mapplot(name='u')

#%%

t = xr.open_dataarray('local/runs/11-14/bias/nov/rf/2012/test.target.nc')
# t.attrs['name'] = 'target'
p = xr.open_dataarray('local/runs/11-14/bias/nov/rf/2012/test.predict.nc')
# p.attrs['name'] = 'predict'
p.name = 'predict'
tp = xr.merge([t, p])
tp

tp['residuals'] = tp.target - tp.predict
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

ds.psy.plot.mapplot(
    share     = ['bounds', 'cticks'],
    ax        = (1, 3),
    tight     = True,
    title     = '%(name)s',
    cmap      = 'viridis',
    bounds    = {'method': 'roundedsym', 'N': 20},
    cticks    = {'method': 'roundedsym', 'N': 3},
    stock_img = True,
    lonlatbox = 'Europe',

)

#%%

def load_predict(path):
    t = xr.open_dataarray(f'{path}/test.target.nc')
    p = xr.open_dataarray(f'{path}/test.predict.nc')
    p.name = 'predict'
    tp = xr.merge([t, p])
    tp['residuals'] = tp.target - tp.predict
    return tp

ds = load_predict('/Volumes/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias/jan/rf/2012')

#%%
vectors = psy.plot.mapvector(
    'icon_grid_demo.nc', name=[['u', 'v']] * 2, projection='robin',
    ax=(1, 2), lonlatbox='Europe')
vectors.plotters[0].update(arrowsize=100)
vectors.plotters[1].update(plot='stream')

#%%
help(tp.psy.plot.mapplot)
psy.plot.mapplot.keys(grouped=True)

psy.plot.mapplot.docs('bounds')
