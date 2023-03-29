%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import numpy  as np
import os
import shap
import xarray as xr

from mlky import *

from sudsaq import  (
    Config as Conf,
    Section as Sect
)
from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)

#%%

def load_from_run(path, kind, objs=['model', 'data', 'target', 'predict']):
    """
    Loads objects from a given run
    """
    files = {
        'model'  : f'{path}/model.pkl',
        'data'   : f'{path}/{kind}.data.nc',
        'target' : f'{path}/{kind}.target.nc',
        'predict': f'{path}/{kind}.predict.nc'
    }
    ret = []

    for obj, file in files.items():
        if obj in objs:
            if os.path.exists(file):
                if file.endswith('.pkl'):
                    ret.append(
                        load_pkl(file)
                    )
                elif file.endswith('.nc'):
                    ret.append(
                        xr.open_dataset(file)
                    )
                else:
                    Logger.error(f'Invalid option: {obj}')
            else:
                Logger.error(f'File not found: {file}')

    return ret

#%

model, data = load_from_run('.local/data/bias/v4.1/jan/2012', 'test', ['model', 'data'])
data = data.stack({'loc': ['lat', 'lon', 'time']}).load()

#%

data = data.head(50)
data

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
    def __init__(self, *args, _dataset=None, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self._dataset = _dataset

    def to_dataset(self):
        """
        """
        if hasattr(self, '_dataset'):
            self._dataset['values'] = (('loc', 'variable'), self.values)
            self._dataset['data']   = (('loc', 'variable'), self.data  )
            self._dataset['base_values'] = self.base_values

            return self._dataset
        else:
            Logger.error('This object is missing the _dataset attribute, did you set it?')

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
            np.array (self['values']),
            np.float_(self['base_values']),
            np.array (self['data']),
            feature_names = self['variable'].values
        )
        ex._dataset = ds[ds.coords.keys()].copy()
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
        _dataset = Dataset(_dataset)
    )

    return explanation

X  = data.to_dataframe().drop(columns=['lat', 'lon', 'time'], errors='ignore')
ex = shap_values(model, X, _dataset=data[data.coords.keys()].copy())


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
