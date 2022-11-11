%load_ext autoreload
%autoreload 2
%matplotlib inline
#%

import numpy  as np
import shap
import xarray as xr

from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)
#%%

model   = load_pkl('local/runs/11-14/bias/nov/rf/2012/model.pkl')
data    = xr.open_dataset('local/runs/11-14/bias/nov/rf/2012/test.data.nc')
target  = xr.open_dataarray('local/runs/11-14/bias/nov/rf/2012/test.target.nc')
predict = xr.open_dataarray('local/runs/11-14/bias/nov/rf/2012/test.predict.nc')
predict.name = 'predict'

data.load(), target.load(), predict.load()

#%

data = data.stack({'loc': ['lat', 'lon', 'time']})
# data = data.transpose('loc', 'variable')
_data = data.head(50)
#%%

from types import SimpleNamespace

Logger = SimpleNamespace(
    exception = lambda string: print(f'EXCEPTION: {string}'),
    info      = lambda string: print(f'INFO: {string}'),
    error     = lambda string: print(f'ERROR: {string}'),
    debug     = lambda string: print(f'DEBUG: {string}')
)

import multiprocessing as mp
import os

from functools import partial
from tqdm import tqdm

def _calc_interactive_values(data, explainer):
    """
    """
    shap    = xr.zeros_like(data)
    shap[:] = explainer.shap_interaction_values(data.values)
    return shap

def _calc_shap_values(data, explainer):
    """
    """
    # shap    = xr.zeros_like(data)
    # shap[:] = explainer.shap_values(data.values)
    # return shap
    return explainer(data)

def shap_values(model, data, n_jobs=-1):
    """
    """
    Logger.debug('Creating explainer')
    explainer = shap.TreeExplainer(model, data)#, feature_perturbation='tree_path_dependent')

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

    return rets

    # Combine the results together to one Explanation object
    explanation = shap.Explanation(
        np.vstack([ret.values      for ret in rets]),
        np.vstack([ret.base_values for ret in rets]),
        np.vstack([ret.data        for ret in rets]),
        feature_names = ret.feature_names
    )

    return explanation

X  = _data.to_dataframe()
_ex = shap_values(model, X)
#%%
_ex[1]
ex[1]

#%%

exp = shap.Explanation(
    np.vstack([ret.values      for ret in ex]),
    np.vstack([ret.base_values for ret in ex]),
    np.vstack([ret.data        for ret in ex]),
    feature_names = X.columns
)
exp


#%%
data
rets

r = rets.unstack()
s = rets.to_array().transpose('loc', 'variable')
s

#%%
#%%
#%%

shap.plots.beeswarm(s.values, plot_size=(10, 5))

s.to_dataframe()
rets.to_dataframe()
r.to_dataframe()


#%%
import seaborn as sns

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

#%%

explainer = shap.Explainer(model)
shap_values = explainer(_data.values)

shap_values

#%%
shap.initjs()
d = _calculate_shap(_data, explainer)

shap.waterfall_plot(shap_values[0])

#%%
help(shap.force_plot)
dir(shap)

shap_values.base_values = shap_values.base_values

help(shap.waterfall_plot)

#%%

X = _data.to_dataset('variable').to_dataframe()
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap.waterfall_plot(shap_values[0])

#%%

# help(shap.plots.heatmap)

shap.plots.heatmap(ex, show=False)

# plt.gcf().set_size_inches(8)

#%%
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10, 5))

# Set plot_size=None to disable auto resizing
# Set show=False to enable control over the plot
shap.summary_plot(exp, X, show=False)
shap.summary_plot(exp, X, show=False)

# Tweak plot as needed
plt.savefig('local/runs/11-14/bias/nov/rf/2012/shap.summary.png')

#%%
ex
shap.dependence_plot(0, ex.values, ex.data, feature_names=ex.feature_names, interaction_index=1)
shap.dependence_plot(0, ex)
#%%

dir(ex)

#%%
#%%
#%%
#%%
#%%
#%%
#%%

def to_explanation(ds):
    """
    """
    assert tuple(ds.dims.keys()) == ('variable', 'loc'), f'Dataset is not of expected shape (variable, loc): {ds.dims}'

    return shap.Explanation(
        ds['values'].values,
        ds['base_values'].values,
        ds['data'].values,
        feature_names = ds['variable'].values
    )

def to_dataset(explanation, data):
    """
    """
    ds = xr.Dataset(coords=data.coords)

    dims = list(ds.dims.values())
    ds['values']      = (('loc', 'variable'), explanation.values     )
    ds['data']        = (('loc', 'variable'), explanation.data       )
    ds['base_values'] = (('loc',           ), explanation.base_values)

    return ds

ds = to_dataset(ex, _data)

#%%

explainer = shap.TreeExplainer(model, X)

help(shap.TreeExplainer)

help(explainer)

#%%
X.shape
X.sample(45)

help(X.sample)

#%%

years = [2012, 2013, 2014, 2015]
{

}
#%%

from glob import glob

runs = glob('/Volumes/MLIA_active_data/data_SUDSAQ/models/2011-2015/bias/*/**/**')

stat = {}
for run in runs:
    split = run.split('/')
    month = split[-3]
    year  = split[-1]

    if month not in stat:
        stat[month] = []

    if year.startswith('20'):
        stat[month].append(int(year))

stat

#%%

from sudsaq.config import Config

c = Config('sudsaq/configs/dev/ml/monthly/create.bias.11-15.yml', 'jan')
c.input.glob

xr.open_mfdataset(c.input.glob)
