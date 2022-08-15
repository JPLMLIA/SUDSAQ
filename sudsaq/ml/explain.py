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

from tqdm import tqdm

from sudsaq.config import Config
from sudsaq.utils  import (
    init,
    load_pkl
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

def heatmap(explanation):
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
    n_jobs  = {-1: os.cpu_count(), None: 1}.get(n_jobs, n_jobs)
    subsets = np.array_split(data, n_jobs)

    Logger.debug(f'Using {n_jobs} jobs')

    bar  = tqdm(total=len(subsets), desc='Processes Finished')
    rets = []

    explainer = shap.TreeExplainer(model, data)
    with mp.Pool(processes=n_jobs) as pool:
        for ret in pool.imap(explainer, subsets):
            rets.append(ret)
            bar.update()

    # Combine the results together to one Explanation object
    explanation = shap.Explanation(
        np.vstack([ret.values      for ret in rets]),
        np.hstack([ret.base_values for ret in rets]),
        np.vstack([ret.data        for ret in rets]),
        feature_names = ret.feature_names
    )

    return explanation

def explain(model=None, data=None, output=None, kind=None):
    """
    """
    config = Config()

    # Verify there is an output directory, otherwise disable outputting
    if not output:
        if config.output.path:
            output = config.output.path
        else:
            Logger.warning(f'No output provided, disabling saving objects')
            for key in config.output:
                config.output[key] = False

    # Use config provided `kind` if available
    kind = config.get('kind', 'input')

    # Load the model from a pickle if provided
    if model is None:
        try:
            model = load_pkl(config.input.model)
        except:
            Logger.exception(f'Failed to load model')
            return 1

    # # Load data if not provided
    # if data is None:
    #     data, target = load(config, split=True, lazy=False)

    if data is None:
        Logger.info(f'Loading data from {config.input.data}')
        data = xr.open_dataset(config.input.data)
        data = data.stack({'loc': ['lat', 'lon', 'time']})
        data = data.load()

    Logger.info('Generating SHAP explanation, this may take awhile')
    explanation = shap_values(model, data.to_dataframe(), n_job=config.n_jobs)

    save_objects(
        output      = output,
        kind        = kind,
        explanation = to_dataset(explanation)
    )

    # Plots
    Logger.info('Generating SHAP plots')
    summary(explanation, data, save=f'{output}/shap.summary.png')
    heatmap(explanation)


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
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())

    state = False
    try:
        state = explain()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
