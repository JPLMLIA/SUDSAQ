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
            _dataset      = self.drop(list(self)).copy()
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

#%%

import xarray as xr

t = xr.open_dataarray('.local/data/oh/test.target.nc').load()
data = t.data.flatten()
#%%
t = xr.open_dataarray('test.target.nc').load()
s = t.stack(stack=['lat', 'lon', 'time'])
l = s.where(s > 1e3, drop=True)
l.size / s.size
#%%

w = t.where(t > 1e3, drop=True)

s = t.stack(stack=['lat', 'lon', 'time'])

l = s.where(s > 1e3, drop=True)

s.size
l.size
l.size / s.size

#%%

l.time.min()
l.time.max()

#%%
t.where(t > 1e3, drop=True).time.size
t.where(t > 1e3, drop=True).mean().data
t.where(t < 1e3, drop=True).mean().data

help(t.where)
t
t.plot.hist()

#%%

from mlky import Config

c = Config('research/james/definitions.yml', 'default<-v6<-2013-16')
c.range

#%%
import numpy as np

w = t.where(t > 1e3, np.nan)
w

#%%


#%%

fig = plt.figure(figsize=(7, 5))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
draw(t.max('time'), ax, vmax=1e6)


#%%


fig, ax = plt.subplots(figsize=(7, 5))
draw(w.max('time'), ax)

t.std()

#%%

t = xr.open_dataarray('.local/data/oh/test.target.nc').load()
p = xr.open_dataarray('.local/data/oh/test.predict.nc').load()
r = t - p

#%%

fig = plt.figure(figsize=(7, 5))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
draw(r.max('time'), ax, vmax=1e6, vmin=0, title="Residuals")

#%%
from mlky   import Section
from mlky.utils import align_print
from yaspin import yaspin

from yaspin.spinners import Spinners

def forspin(func, iterable, args=(), kwargs={}, spinner=None, text=None, data={}):
    """
    Calls func(item, *args, data=data, **kwargs) for item in iterable and
    records any errors. If func returns a string, that string will be counted
    as an error and all errors displayed with their total count.

    Parameters
    ----------
    func: function
        The function to apply to each entry of `iterable`
    iterable: iterable
        Any iterable object
    args: tuple
        Passes to `func(item, *args, **kwargs)`
    kwargs: dict
        Passes to `func(item, *args, **kwargs)`
    spinner: any, defaults=None
        Any yaspin compatible spinner option
    text: str, defaults=None
        Text to use for the spinner
    data: dict
        Format keys to apply to the text. This is done after every iteration so
        updating the values of this dict will reflect in the text. Additional
        data variables will be populated for use include:
            iter - Current iteration
            tota - len(iterable)
            perc - Percentage = iter/tota*100
        This is also passed to the func as a key-word argument


    """
    data['tota'] = len(iterable)
    errors = Section()
    with yaspin(spinner, text=text, reversal=True) as sp:
        for i, item in enumerate(iterable):
            data['iter'] = i
            data['perc'] = i / t * 100

            e = None
            try:
                ret = func(item, *args, data=data, **kwargs)
                if ret is not True:
                    e = ret
            except Exception as e:
                ...
            finally:
                if e:
                    if not errors[e]:
                        errors[e] = 0
                    errors[e] += 1

            if text:
                sp.text = '\n'.join([
                    'Any errors that occur will appear below:',
                    *column_fmt([(f'{v} ({v/t*100:.2f}%)', k) for k, v in errors.items()], print=False),
                    text.format(**data) + ' '
                ])

from datetime import datetime as dtt

class RefreshText:
    def __init__(self, text='', data={}, errors=Section()):
        self.text   = text
        self.data   = data
        self.errors = errors
        self.start  = dtt.now()

    def __str__(self):
        sec = round((dtt.now() - self.start).total_seconds(), 1)
        # Reverse the {errors: count} to ('count (%)', errors) for column_fmt
        reverse = [(f'{v} ({v/t*100:.2f}%)', k) for k, v in self.errors.items()]
        if reverse:
            reverse = column_fmt(reverse, delimiter=':', print=False)

        return '\n'.join([
            *reverse,
            '---------------------------------------------',
            self.text.format(**self.data) + ' '
        ])

def forspin(func, iterable, args=(), kwargs={}, spinner=None, text=None, data={}):
    """
    """
    data['iter'] = 0
    data['perc'] = 0.
    data['tota'] = t = len(iterable)
    errors = {}
    rt = RefreshText(text, data)
    with yaspin(spinner, text=rt, reversal=True):
        for i, item in enumerate(iterable):
            data['iter'] = i
            data['perc'] = i / t * 100

            e = None
            try:
                ret = func(item, *args, data=data, **kwargs)
                if ret is not True:
                    e = ret
            except Exception as e:
                ...
            finally:
                if e:
                    # if not errors[e]:
                    if e not in errors:
                        errors[e] = 0
                    errors[e] += 1

            rt.errors = errors
            rt.data = data

import numpy as np
import time

def pull(v, data):
    time.sleep(.1)
    n = np.random.randint(10)
    if n >= 7:
        return True
    data['failed'] += 1
    return f'Error {n}'

data = {'failed': 0}
forspin(pull, [0]*300,
    spinner = Spinners.moon,
    text    = 'Total: {tota} | Completed: {iter} ({perc}) | Failed: {failed}',
    data    = data
)


#%%
import shap
import xarray as xr

from sudsaq.ml.explain import Dataset

file = '.local/data/shap/v4/jul/2011/test.explanation.nc'
file = '/Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median/jun/**/test.explanation.nc'

ds = Dataset(xr.open_mfdataset(file))


#%%

def cont_to_ex(dir):
    """
    Converts TreeInterpreter outputs to SHAP Explanation objects
    """
    bias = xr.open_mfdataset(f'{dir}/test.bias.nc').bias.data
    data = xr.open_mfdataset(f'{dir}/test.data.nc').to_array('variable').transpose('lat', 'lon', 'time', 'variable').data
    cont = xr.open_mfdataset(f'{dir}/test.contributions.nc').to_array('variable').transpose('lat', 'lon', 'time', 'variable')

    ds = Dataset(cont.to_dataset(name='values'))
    ds['data'] = (('lat', 'lon', 'time', 'variable'), data)
    ds['base_values'] = (('lat', 'lon', 'time'), bias)
    ex = ds.stack(loc=['lat', 'lon', 'time']).transpose().to_explanation()

    return ds, ex

ts, tx = cont_to_ex('/Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median/jun/**')
ts
ts.to_netcdf('/Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median/jun/treeinterpreter.explanations.nc')

#%%
from mlky import Section
from tqdm import tqdm

def load_regions(regions, ds):
    """
    regions: dict
    """
    r = Section()
    for key, region in tqdm(regions.items(), desc='Loading Regions'):
        lat, lon = region
        region = ds.sel(lat=lat).sel(lon=lon).load()
        r[key] = dict(
            data    = region,
            aligned = xr.align(ds, region, join='left')[1],
            bounds  = f'lat: ({lat.start}, {lat.stop}), lon: ({lon.start}, {lon.stop})'
        )

    return r

#%%

# Mar: 1, 2
# Jul: 3, 4, 5
#                     lat, lon
regions = {
    # 1: [slice(15.9, 21.0), slice( 61.0,  68.5)],
    # 2: [slice(34.2, 38.8), slice(121.8, 124.7)],
    3: [slice(31.1, 36.7), slice(-89.2, -81.0)],
    4: [slice(45.6, 52.4), slice(  3.6,  17.5)],
    5: [slice(25.8, 34.3), slice(112.5, 118.6)],
}

#%%

lat, lon = regions[3]
reg = ds.sel(lat=lat).sel(lon=lon).load()

#%%

t = load_regions(regions, ds)

#%%

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

#%%

# Verify region
reg = t[3]
draw(reg.aligned.count(['time', 'variable']).data, title=f'Bounds: {reg.bounds}')

#%%

reg = reg.data.mean(['lat', 'lon', 'time']).to_explanation(auto=False)
# reg = reg.data.to_explanation().mean(0)

shap.plots.waterfall(reg, max_display=20)

#%%

for i, region in t.items():
    reg = region.data.to_explanation(auto=True).mean(0)
    shap.plots.waterfall(reg, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f'shap.waterfalls.jun.region-{i}.png')
    plt.close('all')

#%%

draw(cs.mean('time').sel(variable='momo.ps')['values'])

#%%

_r = r.mean(['lat', 'lon', 'time']).to_explanation()
shap.plots.waterfall(_r, max_display=20)

#%%

_r = ex.mean(['lat', 'lon', 'time']).to_explanation()
shap.plots.waterfall(_r, max_display=20)
#%%

shap.plots.waterfall(mx, max_display=20)


#%%
r2 = r2.mean(['lat', 'lon', 'time']).to_explanation()
shap.plots.waterfall(r2, max_display=20)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
import xarray as xr

from sudsaq.utils import load_from_run

data, target = load_from_run('/Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median/jul/2011', 'test', ['data', 'target'])

data = data.stack({'loc': ['lat', 'lon', 'time']})
data = data.to_array()
target = target.stack({'loc': ['lat', 'lon', 'time']})
target = target['target']

target = target.dropna('loc')
data = data.dropna('loc')

data, target = xr.align(data, target)

clustering = shap.utils.hclust(data.values.T, target.values)

#%%
"""
Often features in datasets are partially or fully redundant with each other. Where redundant means that a model could use either feature and still get
same accuracy. To find these features practitioners will often compute correlation matrices among the features, or use some type of clustering method.
When working with SHAP we recommend a more direct approach that measures feature redundancy through model loss comparisions. The shap.utils.hclust
method can do this and build a hierarchical clustering of the feature by training XGBoost models to predict the outcome for each pair of input
features. For typical tabular dataset this results in much more accurate measures of feature redundancy than you would get from unsupervised methods
like correlation.

Once we compute such a clustering we can then pass it to the bar plot so we can simultainously visualize both the feature redundancy structure and the
feature importances. Note that by default we don’t show all of the clustering structure, but only the parts of the clustering with distance < 0.5.
Distance in the clustering is assumed to be scaled roughly between 0 and 1, where 0 distance means the features perfectly redundant and 1 means they
are completely independent.
"""

shap.plots.bar(ex[0], clustering=clustering, max_display=None, clustering_cutoff=.13)
#%%

shap.plots.bar(ex, max_display=None)

#%%

# shap.plots.heatmap(ex, max_display=None)


ts = ds.mean('time').stack(loc=['lat', 'lon']).transpose()
tx = ts.to_explanation()

shap.summary_plot(ex, max_display=10)

shap.summary_plot(tx)

#%%
ex[:, 0].data.mean()
shap.plots.scatter(ex[:, 'momo.no2'], color=ex[:, 'momo.no'])
shap.plots.scatter(ex[:, 'momo.no'], color=ex[:, 'momo.no2'])

shap.plots.scatter(ex[:, 'momo.2dsfc.CH2O'], xmax=0, color=ex[:, 'momo.co'])
shap.plots.scatter(ex[:, 'momo.t'], xmax=0, color=ex[:, 'momo.co'])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
