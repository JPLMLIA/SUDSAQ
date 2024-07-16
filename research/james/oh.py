"""
Investigates the extreme scores of emulator/OH runs, such as:

    models/emulator/default.gattaca.oh.oh-300/jul/2015
        mape    = 543316032.0
        rmse    = 4956.4189453125
        r2      = 0.9999692780648975
        r corr  = 0.9999846390519256
"""
%matplotlib inline
%cd /Volumes/MLIA_active_data/data_SUDSAQ/models/emulator/default.gattaca.oh.oh-300/jul/2015

#%%

!ls


from sudsaq.utils import load_from_run

data, target = load_from_run('/Volumes/MLIA_active_data/data_SUDSAQ/models/emulator/default.gattaca.oh.oh-300/jul/2015', 'test', objs=['data', 'target'], stack=True, load=True)
target, predict = load_from_run('/Volumes/MLIA_active_data/data_SUDSAQ/models/emulator/default.gattaca.oh.oh-300/jul/2015', 'test', objs=['target', 'predict'], stack=True, load=True)


#%%

target = target.squeeze()
predict = predict.squeeze()
t = target
d = data.transpose()

t.max()
t.min()
t.quantile(.1)
d.where()

s = d.sel(loc=(t.where(t < t.quantile(.1), drop=True))['loc'].data)
t.where(t < t.quantile(.1), drop=True)

u = s.unstack()
u.mean('time')
#%%
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

from mlky import Sect



import numpy as np
#%%
stats = Sect()
quantiles = range(1, 11, 1)
for quantile in quantiles:
    quantile /= 10
    stats[quantile] = {}
    quant = stats[quantile]

    targ = target.where(target < target.quantile(quantile), drop=True)
    pred = predict.sel(loc=targ['loc'].data)

    stats[quantile] = Sect(
        mape   = mean_absolute_percentage_error(targ, pred),
        rmse   = mean_squared_error(targ, pred, squared=False),
        r2     = r2_score(targ, pred),
        r_corr = pearsonr(targ, pred)[0]
    )


#%%
import xarray as xr
xr.DataArray([quantile.rmse for quantile in stats.values()], dims=('quantile', stats.keys()))
xr.DataArray([quantile.rmse for quantile in stats.values()], coords={'quantile': list(stats.keys())}, name='RMSE').plot()
xr.DataArray([quantile.mape for quantile in stats.values()], coords={'quantile': list(stats.keys())}, name='mape').plot(yscale='symlog')
xr.DataArray([quantile.mape for quantile in stats.values()], coords={'quantile': list(stats.keys())}, name='mape').plot(yscale='log')

#%%
binned = target.copy()
binned['loc'] = list(range(binned['loc'].size))
b = binned.groupby_bins('loc', bins=10)
b

predcopy = predict.copy()
predcopy['loc'] = list(range(predict['loc'].size))

#%%
def calc_rmse(bin):
    pred = predcopy.sel(loc=bin['loc'].data)
    mean_squared_error(bin, pred, squared=False)

rmse = b.apply(calc_rmse)

#%%

for interval, bin in b:
    print(bin.min().data, bin.max().data)
