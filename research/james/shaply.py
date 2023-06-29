%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import logging
import numpy  as np
import os
import shap
import fasttreeshap
import xarray as xr

from mlky import *

from sudsaq import  (
    Config as Conf,
    Section as Sect
)
from sudsaq.utils import (
    align_print,
    load_from_run,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)

#%%

logging.basicConfig(level=logging.DEBUG)

#%

model, data = load_from_run('.local/data/bias/v4.1/jan/2012', 'test', ['model', 'data'])
data = data.stack({'loc': ['lat', 'lon', 'time']}).load()

#%%

# data = data.head(int((data['loc'].size) * .1))
data = data.head(1000)
data = data[['momo.t', 'momo.u', 'momo.v']]

#%%

from types import SimpleNamespace
#
Logger = SimpleNamespace(
    exception = lambda string: print(f'EXCEPTION: {string}'),
    info      = lambda string: print(f'INFO: {string}'),
    error     = lambda string: print(f'ERROR: {string}'),
    debug     = lambda string: print(f'DEBUG: {string}')
)
#
import multiprocessing as mp
import os
#
from functools import partial
from tqdm import tqdm


class Explanation(shap.Explanation):
    """
    """
    def __init__(self, values, _dataset=None, **kwargs):
        """
        """
        if issubclass(type(values), fasttreeshap._explanation.Explanation):
            e = values
            super().__init__(
                values        = e.values,
                base_values   = e.base_values,
                data          = e.data,
                **kwargs
            )
        else:
            super().__init__(values, **kwargs)

        if _dataset is not None:
            _dataset['variable'] = self.feature_names
            self._dataset = _dataset

    def to_dataset(self):
        """
        """
        if hasattr(self, '_dataset'):
            self._dataset['values']      = ('loc', 'variable'), self.values
            self._dataset['data']        = ('loc', 'variable'), self.data
            self._dataset['base_values'] = ('loc'            ), self.base_values.flatten()

            return self._dataset
        else:
            Logger.error('This object is missing the _dataset attribute, did you set it?')

#%%
class Dataset(xr.Dataset):
    """
    Small override of xarray.Dataset that enables regex matching names in the variables
    list
    """
    __slots__ = () # Required for subclassing

    def to_explanation(self):
        """
        """
        ex = Explanation(
            values        = np.array (self['values']     ),
            base_values   = np.float_(self['base_values']),
            data          = np.array (self['data']       ),
            feature_names = self['variable'].values,
            _dataset      = data.drop(list(data)).copy()
        )

        return ex


def summary(explanation, data, save=None):
    """
    """
    # fig, ax = plt.subplots(figsize=(10, 5))

    # Set plot_size=None to disable auto resizing
    # Set show=False to enable control over the plot
    shap.summary_plot(explanation, data, show=False if save else True)

    if save:
        Logger.info(f'Saving summary plot to {save}')
        plt.savefig(save)

def heatmap(explanation, save=None):
    """
    """
    shap.plots.heatmap(explanation, show=False if save else True)

    if save:
        Logger.info(f'Saving heatmap plot to {save}')
        plt.savefig(save)

def dependence(explanation, save=None):
    """
    """
    shap.dependence_plot(0, explanation.values, explanation.data, feature_names=explanation.feature_names, interaction_index=1, show=False)

    if save:
        Logger.info(f'Saving dependence plot to {save}')
        plt.savefig(save)

def fast_shap_values(model, data, n_jobs=-1, _dataset=None):
    """
    """
    Logger.debug('Performing FastTreeSHAP calculations')

    explainer   = fasttreeshap.TreeExplainer(model)
    explanation = Explanation(
        explainer(data, check_additivity=False),
        feature_names = data.columns,
        _dataset      = _dataset
    )

    Logger.debug('Finished SHAP calculations')

    return explanation

def shap_values(model, data, n_jobs=-1, _dataset=None):
    """
    """
    Logger.debug('Creating explainer')
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')

    n_jobs  = {-1: os.cpu_count(), None: 1}.get(n_jobs, n_jobs)
    subsets = np.array_split(data, n_jobs)

    Logger.debug(f'Using {n_jobs} jobs')
    Logger.debug('Performing SHAP calculations')

    # Disable additivity check for now, needs more looking into
    # Issues due to multiprocessing/splitting the input X
    func = partial(explainer, check_additivity=False)

    bar  = tqdm(total=len(subsets), desc='Processes Finished')
    rets = []
    with mp.Pool(processes=n_jobs) as pool:
        for ret in pool.imap(func, subsets):
            rets.append(ret)
            bar.update()

    # Combine the results together to one Explanation object
    explanation = Explanation(
        np.vstack([ret.values      for ret in rets]),
        np.  mean([ret.base_values for ret in rets]),
        np.vstack([ret.data        for ret in rets]),
        feature_names = data.columns,
        _dataset = _dataset
    )

    return explanation

#%%
X  = data.to_dataframe().drop(columns=['lat', 'lon', 'time'], errors='ignore')
ex = fast_shap_values(model, X, _dataset=Dataset(data.drop(list(data)).copy()))
ex

#%%
def test_conversions(exp):
    """
    Checks that the conversion functions of Explanation<-->Dataset work
    correctly

    Parameters
    ----------
    exp: Explanation
        A SHAP Explanation object with the `to_dataset` custom function.
        Generally, this should be the return of sudsaq.ml.explain:shap_values
    """
    ds = exp.to_dataset()
    ex = ds.to_explanation()

    assert (ex.values == exp.values).all()
    assert (ex.base_values == exp.base_values).all()
    assert (ex.data == exp.data).all()

    return True

test_conversions(ex)

#%%

class Explanation(shap.Explanation):
    def __init__(self, *args, _dataset=None, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        if _dataset is not None:
            _dataset['variable'] = self.feature_names
            self._dataset = _dataset

    def to_dataset(self):
        """
        """
        if hasattr(self, '_dataset'):
            self._dataset['values']      = ('loc', 'variable'), self.values
            self._dataset['data']        = ('loc', 'variable'), self.data
            self._dataset['base_values'] = ('loc'            ), self.base_values.flatten()

            return self._dataset
        else:
            Logger.error('This object is missing the _dataset attribute, did you set it?')




#%%
values, base_values=, data=, feature_names=
help(shap.Explanation)
e = Explanation([0], [])
e.values
re = ex.values
re
re.
dir(re)

#%%


for k in dir(re):
    print(k, type(getattr(re, k)))#, getattr(re, k))

#%%

from sudsaq.utils import save_netcdf

ds = ex.to_dataset()
save_netcdf(ds, 'explanation', '.local/data/shap/shap.explanation.nc')
_ex, = load_from_run('.local/data/shap/', objs=['explanation'])

_ex.load()

_ex = Dataset(_ex)
ex_ = _ex.to_explanation()
ex_

ex

#%%
#%%
ns
ds
ns.stack({'loc': ['lat', 'lon', 'time']})

#%%


import cf_xarray as cfxr

def encode(data):
    """
    """
    return cfxr.encode_multi_index_as_compress(data, 'loc')

def decode(file):
    """
    """
    ds = xr.open_dataset(file)
    return cfxr.decode_compress_to_multi_index(ds, 'loc')

file = '.local/data/shap/small_ex.nc'
enc = encode(ds)
enc.to_netcdf(file, engine='netcdf4')
ns = decode(file)
ns.cf


#%%
#%%
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%
#%%
#%% Gattaca terminal
import matplotlib
matplotlib.use("module://imgcat")
import xarray as xr
from sudsaq.ml.explain import to_explanation
from sudsaq.utils import load_from_run
#
model, data = load_from_run('/scratch_lg_edge/sudsaq/models/toar/local/mean/v4/dec/2011', 'test', ['model', 'data'])
unst = data.unstack()
data = unst.resample(time='3D').mean()
data = data.where(data.time != data.time[-1], drop=True)
data = data.stack({'loc': ['lat', 'lon', 'time']})
#
ds = xr.open_dataset('test.explanation.nc')
ds = ds.stack({'loc': ['lat', 'lon', 'time']})
ex = to_explanation(ds)
ex.base_values = ex.base_values.reshape(1, 512000)
#
import shap

ex.data = ex.data.T
ex.values = ex.values.T
# ex.base_values = ex.base_values.T

n = 110

plt.close('all')
shap.plots.bar(ex, max_display=n, show=False)
plt.title('Average Absolute Impact on Prediction Value')
plt.tight_layout()
plt.savefig('shap.bar.png')

plt.close('all')
shap.summary_plot(ex, max_display=n, show=False)
plt.title('Feature Value Impact on Prediction Value')
plt.tight_layout()
plt.savefig('shap.summary.png')

#%%
#%%
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#%%
#%%
#%% SUDSAQ Explorer UI
import matplotlib.pyplot as plt

def dependence(feat1, feat2='auto'):
    return shap.dependence_plot(
        feat1,
        shap_values        = ex.values,
        features           = ex.data,
        feature_names      = ex.feature_names,
        display_features   = None,
        interaction_index  = feat2,
        color              = '#1E88E5',
        axis_color         = '#333333',
        cmap               = None,
        dot_size           = 16,
        x_jitter           = 0,
        alpha              = 1,
        title              = None,
        xmin               = None,
        xmax               = None,
        ax                 = None,
        show               = False
    )

#%%


dependence('momo.co')
plt.show()

#%%
plt.close('all')
shap.summary_plot(exp, max_display=20, show=True)
plt.tight_layout()
plt.savefig('.local/data/toar.local.v4.dec/summary.png')

#%%

shap.plots.heatmap(ex)

#%%

shap.plots.waterfall(ex[0])

shap.waterfall_plot(ex[0])

#%%

exp = shap.Explanation(ex.values, ex.base_values[0][0], ex.data, feature_names=ex.feature_names)
shap.plots.waterfall(exp[0], show=False)
plt.tight_layout()
plt.savefig('.local/data/toar.local.v4.dec/waterfall.0.png')

#%%%%%%%%

Logger.info('Resampling data')
resample(ds, config.explain.resample)

#%%
data = unst.resample(time='3D')

data = getattr(data, how)()
if

data

def resample(ds, time=):
    if config.explain.resample:

        unst = data.unstack()

        # Remove the last timestamp if it was an incomplete group
        data = unst.resample(time='3D').mean()
        if unst.time.size % 3:
            data = data.where(data.time != data.time[-1], drop=True)

        data = data.stack({'loc': ['lat', 'lon', 'time']})
        return flatten(data)
        Logger.debug(f'Resampled to 3D: {data.dims}')

#%%
data = data.sortby('time')
unst = data.unstack()
help(data.resample)

#%%
res = unst.resample(time='3D')
d = getattr(res, 'mean')()

d
size = data.time.size
data

#%%

data = data.unstack().resample(time=freq)
data = getattr(data, how)()
data = flatten(data)

#%%
from sudsaq.data import flatten

def unstacked(func):
    """
    Unstacks the first parameter of the decorated `func` and restacks it if
    the incoming `data` is already stacked, otherwise do nothing
    """
    def wrapped(data, *args, **kwargs):
        loc = False
        if 'loc' in data.dims:
            print('Unstacking')
            loc  = True
            data = data.unstack()

        data = func(data, *args, **kwargs)

        if loc and isinstance(data, (
            xr.core.dataarray.DataArray,
            xr.core.dataarray.Dataset
        )):
            return flatten(data)
        return data

    return wrapped

@unstacked
def subsample(data, dim, N):
    """
    Subsamples along a dimension by dropping every N sample

    Parameters
    ----------
    data: xarray
        Data to subsample on
    dim: str
        Name of the dimension to subsample
    N: int
        Every Nth sample is dropped
    """
    # Select every Nth index
    drop = data[dim][N-1::N]
    return data.drop_sel(**{dim: drop})

ss = subsample(data, 'time', 2)

ss

d = data.unstack('loc')
d = data.to_dataset(name='abc')
d = data.to_array()
'loc' in d.dims
d

#%%%%%%%%%%

import logging
logging.basicConfig(level=logging.DEBUG)
Logger = logging.getLogger('interpreter')
from dask.distributed import Client
import os
client = Client(n_workers=os.cpu_count()-1, threads_per_worker=1)
client
from sudsaq import Config
config = Config('definitions.yml', 'gattaca<-v4<-bias[median]<-apr')
from wcmatch import glob as _glob
glob = lambda pattern: _glob.glob(pattern, flags=_glob.BRACE)
files = []
for string in config.input.glob:
    match = glob(string)
    Logger.debug(f'Collected {len(match)} files using "{string}"')
    files += match

import xarray as xr
ds = xr.open_mfdataset(files, engine='netcdf4', lock=False)

from sudsaq.data import daily, Dataset, config_sel
config.input.daily.momo.local = False
ds = Dataset(ds)
ds = config_sel(ds, config.input.sel)

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

import xarray as xr
from glob import glob
from sudsaq.data import Dataset
v4 = '[^\.]+\.(
?!hno3|oh|pan|q2|sens|so2|T2|taugxs|taugys|taux|tauy|twpc|2dsfc.CFC11|2dsfc.CFC113|2dsfc.CFC12|ch2o|cumf0|2dsfc.dms|2dsfc.HCFC22|2dsfc.H1211|2dsfc.H1301|2dsfc.mc.oc|2dsfc.dflx.bc|2dsfc.dflx.oc|2dsfc.LR.SO2|2dsfc.CH3COOOH|prcpl|2dsfc.C3H7OOH|dqlsc|2dsfc.mc.pm25.salt|2dsfc.CH3COO2|u10|2dsfc.dflx.nh4|2dsfc.mc.nh4|2dsfc.dflx.dust|2dsfc.mc.pm25.dust|osr|osrc|ssrc|v10|2dsfc.OCS|2dsfc.taut|ccoverl|ccoverm|2dsfc.DCDT.HOX|2dsfc.DCDT.OY|2dsfc.DCDT.SO2|slrdc|uvabs|dqcum|dqdad|dqdyn|dqvdf|dtdad|cumf|ccoverh|prcpc|2dsfc.BrCl|2dsfc.Br2|dtcum|2dsfc.mc.sulf|2dsfc.HOBr|dtlsc|2dsfc.Cl2|2dsfc.CH3CCl3|2dsfc.CH3Br|2dsfc.ONMV|2dsfc.MACROOH|2dsfc.MACR|2dsfc.HBr|Restart|agcm|CHEMTMP|SYSIN|gralb|.*gt3|stderr|tcr2).*'
strings = [
  '/data/MLIA_active_data/data_SUDSAQ/data/momo/201[1-5]/01.nc',
  '/data/MLIA_active_data/data_SUDSAQ/data/toar/matched/201[1-5]/01.nc',
  '/data/MLIA_active_data/data_SUDSAQ/data/gee/modis/*.nc',
  '/data/MLIA_active_data/data_SUDSAQ/data/gee/pop_2010_fixed.nc'
]
for path in strings:
    ds = xr.open_mfdataset(path, lock=False)
    ds.load()
    del ds

xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/data/gee/modis/*.nc', lock=False).load()
xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/data/gee/pop_2010_fixed.nc', lock=False)

xr.open_mfdataset('/data/MLIA_active_data/data_SUDSAQ/data/gee/modis/*.nc', lock=False).load()
/projects/mlia-active-data/data_SUDSAQ/data/momo/20{0[5-9],1[0-5]}/04.nc
/projects/mlia-active-data/data_SUDSAQ/data/toar/matched/20{0[5-9],1[0-5]}/04.nc

#%%

import xarray as xr
ds = xr.open_mfdataset('*.nc').load()
for year, yds in ds.groupby('time.year'):
    yds.to_netcdf(f'xr.{year}.nc')

#%%
import xarray as xr
f = []
for p in ['/data/MLIA_active_data/data_SUDSAQ/data/gee/modis/*.nc', '/data/MLIA_active_data/data_SUDSAQ/data/gee/pop_2010_fixed.nc']: f += glob(p)

xr.open_mfdataset(f, lock=False).load()

#%%
paths = ["/projects/mlia-active-data/data_SUDSAQ/data/momo/201[1-5]/02.nc",
"/projects/mlia-active-data/data_SUDSAQ/data/toar/matched/201[1-5]/02.nc",
"/projects/mlia-active-data/data_SUDSAQ/data/gee/modis/*.nc",
"/projects/mlia-active-data/data_SUDSAQ/data/gee/pop.2010.nc"]
for p in paths: print(glob(p))

from glob import glob as g0
from wcmatch import glob as g1
g2 = lambda pattern: g1.glob(pattern, flags=g1.BRACE)
p = "/projects/mlia-active-data/data_SUDSAQ/data/momo/201[1-5]/02.nc"
g2(p)
