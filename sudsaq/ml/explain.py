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
from tqdm      import tqdm
from glob      import glob

from sudsaq.config import Config
from sudsaq.utils  import (
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

Logger = logging.getLogger('sudsaq/ml/explain.py')

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

def to_explanation(ds):
    """
    Converts an xarray.Dataset to a shap.Explanation
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
    Converts an shap.Explanation to a xarray.Dataset
    """
    ds = xr.Dataset(coords=data.coords)

    dims = list(ds.dims.values())
    ds['values']      = (('loc', 'variable'), explanation.values     )
    ds['data']        = (('loc', 'variable'), explanation.data       )
    ds['base_values'] = (('loc',           ), explanation.base_values)

    return ds

def shap_values(model, data, n_jobs=-1):
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
    explanation = shap.Explanation(
        np.vstack([ret.values      for ret in rets]),
        np.vstack([ret.base_values for ret in rets]),
        np.vstack([ret.data        for ret in rets]),
        feature_names = ret.feature_names
    )

    return explanation

def explain(model, data, kind='test', output=None):
    """
    """
    config = Config()

    Logger.info('Generating SHAP explanation, this may take awhile')
    X = data.to_dataframe().drop(columns=['lat', 'lon', 'time'], errors='ignore')
    explanation = shap_values(model, X, n_jobs=config.get('n_job', -1))

    save_objects(
        output      = output,
        kind        = kind,
        explanation = to_dataset(explanation, data)
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
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'explain',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use. This is likely the same section as given to create.py.'
    )
    parser.add_argument('-k', '--kind',     type     = str,
                                            default  = 'test',
                                            metavar  = '[kind]',
                                            help     = 'Which kind of data to process on, eg. `train`, `test`'
    )

    args, config = init(parser.parse_args())

    folds = glob(f'{config.output.path}/[0-9]*')
    Logger.info(f'Running SHAP calculations for {len(folds)} folds')

    states = 0
    for fold in folds:
        try:
            model, data = load_from_run(fold, args.kind, ['model', 'data'])
            data = data.stack({'loc': ['lat', 'lon', 'time']}).load()
            try:
                ret = explain(model, data, args.kind, fold)
                if isinstance(ret, shap._explanation.Explanation):
                    states += 1
            except Exception:
                Logger.exception('Caught an exception during runtime')
        except Exception:
            Logger.exception(f'Caught an exception during runtime for fold {fold}')

    Logger.info('Finished {states}/{len(folds)} folds successfully')
