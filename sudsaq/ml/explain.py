"""
"""
import argparse
import logging
import matplotlib.pyplot as plt
import multiprocessing   as mp
import numpy             as np
import os
import seaborn           as sns
import shap
import xarray            as xr

from functools import partial
from pathlib   import Path
from tqdm      import tqdm

from sudsaq import  Config
from sudsaq.utils  import (
    init,
    load_from_run,
    save_objects
)

try:
    import fasttreeshap
except:
    fasttreeshap = None

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

Logger = logging.getLogger('sudsaq/ml/explain.py')


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


class Dataset(xr.Dataset):
    """
    Small override of xarray.Dataset that enables regex matching names in the variables
    list
    """
    __slots__ = () # Required for subclassing

    def to_explanation(self, auto=False, stack={'loc': ['lat', 'lon', 'time']}, transpose=False):
        """
        """
        if auto:
            if 'loc' not in self.coords:
                self = self.stack(**stack)
                transpose = True

            if transpose:
                self = self.transpose()

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
        explainer(data),
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
    total   = len(subsets)

    Logger.debug(f'Using {n_jobs} jobs')
    Logger.debug(f'Split into {total} subsets')
    Logger.debug('Performing SHAP calculations')

    # Disable additivity check for now, needs more looking into
    # Issues due to multiprocessing/splitting the input X
    func = partial(explainer, check_additivity=False)

    step = int(total * .1)
    bar  = tqdm(total=total, desc='Processes Finished')
    rets = []
    with mp.Pool(processes=n_jobs) as pool:
        for ret in pool.imap(func, subsets):
            rets.append(ret)
            bar.update()

            if bar.n % step == 0:
                Logger.debug(str(bar))

    # Combine the results together to one Explanation object
    explanation = shap.Explanation(
        np.vstack([ret.values      for ret in rets]),
        np.  mean([ret.base_values for ret in rets]),
        np.vstack([ret.data        for ret in rets]),
        feature_names = data.columns,
        _dataset = _dataset
    )

    return explanation

def explain(model, data, kind='test', output=None):
    """
    TODO: `data` only supports the kind returned by utils.load_from_run, but
    this is different than if it were passed by the pipeline during runtime so
    that needs support
    """
    config = Config()

    # TODO: Support other resampling
    if config.explain.resample:
        Logger.info('Resampling data')
        unst = data.unstack()

        # Remove the last timestamp if it was an incomplete group
        data = unst.resample(time='3D').mean()
        if unst.time.size % 3:
            data = data.where(data.time != data.time[-1], drop=True)

        data = data.stack({'loc': ['lat', 'lon', 'time']})
        Logger.debug(f'Resampled to 3D: {data.dims}')

    Logger.info('Generating SHAP explanation, this may take awhile')
    X = data.to_dataframe().drop(columns=['lat', 'lon', 'time'], errors='ignore')
    if fasttreeshap is not None:
        explanation = fast_shap_values(model, X,
            _dataset = Dataset(data.drop(list(data)).copy())
        )
    else:
        explanation = shap_values(model, X,
            n_jobs   = config.get('n_job', -1),
            _dataset = data[data.coords.keys()].copy()
        )

    save_objects(
        output      = output,
        kind        = kind,
        explanation = explanation.to_dataset()
    )

    if config.output.plots:
        Logger.info('Generating SHAP plots')
        summary(explanation, X, save=f'{output}/shap.summary.png')
        # heatmap(explanation,    save=f'{output}/shap.heatmap.png')

    return explanation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-p', '--patch',    nargs    = '?',
                                            metavar  = 'sect1 ... sectN',
                                            help     = 'Patch sections together starting from sect1 to sectN'
    )
    parser.add_argument('-i', '--inherit',  nargs    = '?',
                                            metavar  = 'sect1 sect2',
                                            help     = 'Order of keys to apply inheritance where rightmost takes precedence over left'
    )
    parser.add_argument('-k', '--kind',     type     = str,
                                            default  = 'test',
                                            metavar  = '[kind]',
                                            help     = 'Which kind of data to process on, eg. `train`, `test`'
    )

    args, config = init(parser.parse_args())

    folds = list(Path(config.output.path).glob('[0-9]*'))
    if not folds:
        Logger.error(f'No folds found for path: {config.output.path}/[0-9]*')
    else:
        Logger.info(f'Running SHAP calculations for {len(folds)} folds')

        success = 0
        for fold in folds:
            try:
                model, data = load_from_run(fold, args.kind, ['model', 'data'])
                data = data.stack({'loc': ['lat', 'lon', 'time']}).load()
                try:
                    ret = explain(model, data, args.kind, fold)
                    if isinstance(ret, shap._explanation.Explanation):
                        success += 1
                except Exception:
                    Logger.exception(f'Caught an exception explaining fold {fold}')
            except Exception:
                Logger.exception(f'Caught an exception loading data for fold {fold}')

        Logger.info(f'Finished {success}/{len(folds)} folds successfully')
