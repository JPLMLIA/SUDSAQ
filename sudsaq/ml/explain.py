"""
"""
# Builtin
import argparse
import logging
import multiprocessing as mp
import os

from functools import partial
from pathlib   import Path

# External
import numpy   as np
import seaborn as sns
import shap
import xarray  as xr

from matplotlib import pyplot as plt
from mlky       import Config
from tqdm       import tqdm

try:
    import fasttreeshap
except:
    fasttreeshap = None

# Internal
from sudsaq.utils import (
    init,
    load_from_run,
    save_objects
)

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

Logger = logging.getLogger('sudsaq/ml/explain')


class Explanation(shap.Explanation):
    """
    Converts a SHAP Explanation to a SUDSAQ Dataset
    """
    def __init__(self, values, _dataset=None, **kwargs):
        if fasttreeshap and issubclass(type(values), fasttreeshap._explanation.Explanation):
                e = values
                super().__init__(
                    values      = e.values,
                    base_values = e.base_values,
                    data        = e.data,
                    **kwargs
                )
        else:
            super().__init__(values, **kwargs)

        if _dataset is not None:
            _dataset['variable'] = self.feature_names
            self._dataset = _dataset

    def to_dataset(self):
        """
        Converts this explanation object to an xarray Dataset object
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
    Converts a SUDSAQ Dataset to a SHAP Explanation
    """
    __slots__ = () # Required for subclassing

    def to_explanation(self, auto=False, stack={'loc': ['lat', 'lon', 'time']}, transpose=False):
        """
        Converts this dataset to a SHAP Explanation object
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
    Generate and optionally save a SHAP summary plot.

    Parameters
    ----------
    explanation : shap.Explanation
        The SHAP explanation object containing SHAP values and other related data.
    data : pandas.DataFrame or numpy.ndarray
        The dataset used to generate the SHAP values.
    save : str, optional, default=None
        The file path to save the plot. If None, the plot is displayed instead of being saved

    Returns
    -------
    None
        The function saves or displays the plot, but does not return any value.

    Examples
    --------
    >>> summary(explanation, data)
    >>> summary(explanation, data, save='summary_plot.png')
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
    Generate and optionally save a SHAP heatmap plot.

    Parameters
    ----------
    explanation : shap.Explanation
        The SHAP explanation object containing SHAP values and other related data.
    save : str, optional, default=None
        The file path to save the plot. If None, the plot is displayed instead of being saved.

    Returns
    -------
    None
        The function saves or displays the plot, but does not return any value.

    Examples
    --------
    >>> heatmap(explanation)
    >>> heatmap(explanation, save='heatmap_plot.png')
    """
    shap.plots.heatmap(explanation, show=False if save else True)

    if save:
        Logger.info(f'Saving heatmap plot to {save}')
        plt.savefig(save)


def dependence(explanation, save=None):
    """
    Generate and optionally save a SHAP dependence plot for the first feature.

    Parameters
    ----------
    explanation : shap.Explanation
        The SHAP explanation object containing SHAP values and other related data.
    save : str, optional, default=None
        The file path to save the plot. If None, the plot is displayed instead of being saved.

    Returns
    -------
    None
        The function saves or displays the plot, but does not return any value.

    Examples
    --------
    >>> dependence(explanation)
    >>> dependence(explanation, save='dependence_plot.png')
    """
    shap.dependence_plot(0, explanation.values, explanation.data, feature_names=explanation.feature_names, interaction_index=1, show=False)

    if save:
        Logger.info(f'Saving dependence plot to {save}')
        plt.savefig(save)


def fast_shap_values(model, data, n_jobs=-1, _dataset=None):
    """
    Calculate SHAP values quickly using FastTreeSHAP.

    Parameters
    ----------
    model : object
        The tree model to explain
    data : pandas.DataFrame
        The dataset for which SHAP values are to be calculated
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1, which means use all processors
    _dataset : optional
        Additional dataset information used to cast back to SUDSAQ-compatible xarray object

    Returns
    -------
    Explanation
        An object containing SHAP values, feature names, and the dataset.
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


def split_shap_values(model, data, n_jobs=-1, _dataset=None):
    """
    Calculate SHAP values by splitting the dataset for parallel processing.

    Experimental.

    Parameters
    ----------
    model : object
        The tree model to explain
    data : pandas.DataFrame
        The dataset for which SHAP values are to be calculated
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1, which means use all processors
    _dataset : optional
        Additional dataset information used to cast back to SUDSAQ-compatible xarray object

    Returns
    -------
    Explanation
        An object containing combined SHAP values from all subsets, feature names, and the dataset.
    """
    Logger.debug('Performing TreeSHAP calculations using method: split')
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


def approx_shap_values(model, data, _dataset=None):
    """
    Calculate approximate SHAP values.

    Parameters
    ----------
    model : object
        The tree model to explain
    data : pandas.DataFrame
        The dataset for which SHAP values are to be calculated
    _dataset : optional
        Additional dataset information used to cast back to SUDSAQ-compatible xarray object

    Returns
    -------
    Explanation
        An object containing approximate SHAP values, feature names, and the dataset.
    """
    Logger.debug('Performing TreeSHAP calculations using method: approx')
    Logger.debug('Creating explainer')
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')

    Logger.debug('Performing SHAP calculations')
    explanation = Explanation(
        explainer.shap_values(data, approximate=True),
        feature_names = data.columns,
        _dataset      = _dataset
    )

    return explanation


def explain(model, data, kind='test', output=None):
    """
    Generate SHAP explanations for a given model and dataset.

    This function supports resampling the data if specified in the configuration,
    and generates SHAP explanations using either FastTreeSHAP or an approximate
    method depending on availability. The results can be saved and optionally
    plotted.

    Parameters
    ----------
    model : object
        The machine learning model to explain.
    data : xarray.DataArray
        The dataset for which SHAP values are to be calculated. Currently supports
        the kind returned by `utils.load_from_run`.
    kind : str, optional
        The name of the dataset being explained.
    output : str, optional
        The directory path where the results and plots will be saved. Default is None.

    Returns
    -------
    Explanation
        An object containing SHAP values, feature names, and the dataset.

    Notes
    -----
    - The `data` parameter currently only supports the format returned by
      `utils.load_from_run`. This is different from the format passed during
      runtime in the pipeline, which needs to be supported in future versions.
    - If `config.explain.resample` is True, the data will be resampled to 3-day
      intervals, and incomplete groups will be removed.

    See Also
    --------
    fast_shap_values : Function to calculate SHAP values using FastTreeSHAP.
    approx_shap_values : Function to calculate approximate SHAP values.
    utils.load_from_run : Utility function to load data in the supported format.
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

    if config.shap == 'fast' and fasttreeshap is not None:
        explanation = fast_shap_values(model, X,
            _dataset = Dataset(data.drop(list(data)).copy())
        )
    elif config.shap == 'split':
        explanation = split_shap_values(model, X,
            n_jobs   = config.get('n_job', -1),
            _dataset = Dataset(data.drop(list(data)).copy())
        )
    else:
        explanation = approx_shap_values(model, X,
            _dataset = Dataset(data.drop(list(data)).copy())
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
