"""
"""
import argparse
import logging
import numpy  as np
import pandas as pd
import xarray as xr

from scipy.stats        import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from tqdm import tqdm

from sudsaq import  (
    Config,
    Section,
    Null
)
from sudsaq.data  import (
    flatten,
    load
)
from sudsaq.ml    import plots
from sudsaq.ml    import treeinterpreter as ti
from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_objects
)

Logger = logging.getLogger('sudsaq/ml/analyze.py')


def perm_importance(model, data, target, output=None):
    """
    """
    config = Config()

    # Make sure the inputs are aligned first
    data, target = xr.align(data, target)

    permimp = permutation_importance(model, data, target, **config.permutation_importance)
    # Only want the summaries, remove the importances array
    del permimp['importances']

    # Convert to a DataFrame and sort by importance value
    df = pd.DataFrame(permimp.values(), columns=data['variable'], index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)

    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    Logger.info('Permutation importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    if output:
        with open(output, 'a') as file:
            file.write('\n\nPermutation importance +/- stddev:\n')
            file.write('\n'.join(strings))
        df.to_hdf(output.replace('.txt', '.h5'), 'permutation')

    return df

def model_importance(model, variables, output=None):
    """
    Retrieves and formats the importances from a RandomForest model
    """
    # Retrieve the importances list and the stddev of these
    imports = model.feature_importances_
    stddev  = np.std([est.feature_importances_ for est in model.estimators_], axis=0)

    # Place into a DataFrame for easier handling
    df = pd.DataFrame(np.array([imports, stddev]), columns=variables, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)

    fmt = {}
    for var, vals in df.items():
        fmt[var] = f'{vals.importance} +/- {vals.stddev}'

    Logger.info('Feature importance +/- stddev:')
    strings = align_print(fmt, enum=True, print=Logger.info)
    if output:
        with open(output, 'w') as file:
            file.write('Feature importance +/- stddev:\n')
            file.write('\n'.join(strings))
        df.to_hdf(output.replace('.txt', '.h5'), 'model')

    return df

def pbc(model, data):
    """
    Predict, Bias, Contributions calculations
    """
    config = Config()

    # Only predict specified regions
    regions = []
    if config.predict_regions:
        unstacked = data.unstack()
        for region, bounds in config.predict_regions.items():
            Logger.debug(f'Selecting region {region} using bounds: lat={bounds.lat}, lon={bounds.lon}')
            regions.append(
                unstacked.sel(
                    lat = slice(*bounds.lat),
                    lon = slice(*bounds.lon)
                ).to_dataset('variable')
            )

        data = flatten(xr.merge(regions)).dropna('loc').transpose('loc', 'variable')

    Logger.info('Predicting using TreeInterpreter')
    if config.split_predict:
        Logger.debug(f'Using {config.split_predict} splits for prediction')
        predicts = []
        biases   = []
        contribs = []

        for split in tqdm(np.split(data, config.split_predict), desc='Processed Splits'):
            predict       = xr.zeros_like(split.isel(variable=0).drop_vars('variable'))
            bias          = xr.zeros_like(predict)
            contributions = xr.zeros_like(split)

            _predict, bias[:], contributions[:] = ti.predict(model, split, **config.treeinterpreter)

            predict[:] = _predict.flatten()

            predicts.append(predict)
            biases.append(bias)
            contribs.append(contributions)

        Logger.debug('Combining splits')
        predict       = xr.concat(predicts, 'loc')
        bias          = xr.concat(biases, 'loc')
        contributions = xr.concat(contribs, 'loc')
    else:
        predict       = xr.zeros_like(data.isel(variable=0).drop_vars('variable'))
        bias          = xr.zeros_like(predict)
        contributions = xr.zeros_like(data)

        _predict, bias[:], contributions[:] = ti.predict(model, data, **config.treeinterpreter)

        predict[:] = _predict.flatten()

    return predict, bias, contributions

def analyze(model=None, data=None, target=None, kind='input', output=None):
    """
    Analyzes a model using different metrics and plots

    Parameters
    ----------
    model: sklearn.ensemble
        A fitted sklearn.ensemble model
    data: xarray.DataArray
        The input data, X
    target: xarray.DataArray
        The input target, y
    kind: str, default = 'default'
        The kind of data this is, eg. Train, Test
        Used for titling
    output: str, default = None
        Directory path to output to
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

    if output:
        mkdir(output)

    # Load the model from a pickle if provided
    if model is None:
        try:
            model = load_pkl(config.input.model)
        except:
            Logger.exception(f'Failed to load model')
            return 1

    # Load data if not provided
    if data is None:
        data, target = load(config, split=True, lazy=False)

    # Stores information about this analysis
    stats = Section('scores', {})

    # Prepare the storage variables
    bias, contributions = None, None
    if not config.not_ti and 'Forest' in str(model):
        predict, bias, contributions = pbc(model, data)
    else:
        Logger.info('Predicting')
        predict    = xr.zeros_like(data.isel(variable=0).drop_vars('variable'))
        predict[:] = model.predict(data.values)
    stats.predict = predict

    if target is not None:
        # Some functions require the target and predict be the same shape
        target, aligned = xr.align(target, predict)
        stats.aligned = aligned

        Logger.info('Calculating scores')
        stats['mape']   = mean_absolute_percentage_error(target, aligned)
        stats['rmse']   = mean_squared_error(target, aligned, squared=False)
        stats['r2']     = r2_score(target, aligned)
        stats['r corr'] = pearsonr(target, aligned)[0]

        # Log the scores
        scores = align_print(stats, enum=False, prepend='  ', print=Logger.info)
        if config.output.scores:
            Logger.info(f'Saving scores to {output}/{kind}.scores.txt')
            with open(f'{output}/{kind}.scores.txt', 'w') as file:
                file.write('Scores:\n')
                file.write('\n'.join(scores))

    # Feature importances
    impout = None
    if config.output.importance:
        impout = f'{output}/{kind}.importance.txt'
        Logger.info(f'Saving importances to {impout}')
    if 'Forest' in str(model):
        stats.mimportance = model_importance(model, data['variable'], output=impout)
    if config.permutation_importance:
        try:
            stats.pimportance = perm_importance(model, data, target, output=impout)
        except:
            Logger.exception('Failed to generate permutation importance')

    # Create plots if enabled
    if config.output.plots:
        if config.plots.importance is not False:
            if stats.imports is not Null:
                plots.importance(
                    df   = stats.mimportance,
                    pdf  = stats.pimportance,
                    save = f'{output}/{kind}.importances.png'
                )

        if target is not None:
            if config.plots.compare_target_predict is not False:
                plots.compare_target_predict(
                    target.unstack().mean('time').sortby('lon'),
                    predict.unstack().mean('time').sortby('lon'),
                    reindex = config._reindex,
                    title   = f'{kind.capitalize()} Set Performance',
                    save    = f'{output}/{kind}.compare_target_predict.png'
                )

            if config.plots.truth_vs_predicted is not False:
                plots.truth_vs_predicted(
                    target.dropna('loc'),
                    aligned.dropna('loc'),
                    label = '\n'.join([score.lstrip()[:15] for score in scores]),
                    save  =  f'{output}/{kind}.truth_vs_predicted.png'
                )

    # Save objects as requested
    save_objects(
        output  = output,
        kind    = kind,
        predict = predict,
        bias    = bias,
        contributions = contributions,
    )

    return stats
