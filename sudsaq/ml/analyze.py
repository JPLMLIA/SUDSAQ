"""
"""
# Builtin
import argparse
import logging
import sys

from glob import glob

# External
import numpy  as np
import pandas as pd
import xarray as xr

from mlky import (
    Config,
    Null,
    Sect
)
from scipy.stats        import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from tqdm import tqdm

# Internal
from sudsaq.data  import (
    flatten,
    Loader
)
from sudsaq.ml    import plots
from sudsaq.ml    import treeinterpreter as ti
from sudsaq.utils import (
    align_print,
    init,
    load_from_run,
    load_pkl,
    mkdir,
    save_objects
)


Logger = logging.getLogger('sudsaq/ml/analyze.py')


def perm_importance(model, data, target, output=None):
    """
    """
    # Make sure the inputs are aligned first
    data, target = xr.align(data, target)

    permimp = permutation_importance(model, data, target, **Config.permutation_importance)

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


def predictTI(model, data, *args, **kwargs):
    """
    Calls and returns TreeInterpreter.predict(model, X)

    Parameters
    ----------
    model: ...
        A compatible tree model with TreeInterpreter
    data: xr.DataArray
        2D data to predict with

    Returns
    -------
    (pred, bias, cont): tuple
        The calculated predictions, biases, and contributions objects from
        TreeInterpreter
    """
    pred = data.isel(variable=0).drop('variable')
    bias = pred.copy()
    cont = data.copy()

    p, b, c = ti.predict(model, data, *args, **kwargs, **Config.treeinterpreter)
    pred[:] = p.ravel()
    bias[:] = b
    cont[:] = c

    return pred, bias, cont


def pbc(model, data):
    """
    Prepares
    """
    # Only predict specified regions
    regions = []
    if Config.predict_regions:
        unstacked = data.unstack()
        for region, bounds in Config.predict_regions.items():
            Logger.debug(f'Selecting region {region} using bounds: lat={bounds.lat}, lon={bounds.lon}')
            regions.append(
                unstacked.sel(
                    lat = slice(*bounds.lat),
                    lon = slice(*bounds.lon)
                ).to_dataset('variable')
            )

        data = flatten(xr.merge(regions)).dropna('loc').transpose('loc', 'variable')

    Logger.info('Predicting using TreeInterpreter')
    if Config.split_predict:
        Logger.debug(f'Using {Config.split_predict} splits for prediction')
        pred = []
        bias = []
        cont = []

        for split in tqdm(np.split(data, Config.split_predict), desc='Processed Splits'):
            p, b, c = predictTI(model, split)
            pred.append(p)
            bias.append(b)
            cont.append(c)

        Logger.debug('Combining splits')
        combine = lambda a: xr.concat(a, 'loc')

        return combine(pred), combine(bias), combine(cont)

    # Data not split, return on the full set
    return predictTI(model, data)


def quantilePredict(model, data, output=None, kind=None):
    """
    Performs extra predictions for RandomForestQuantileRegressor models
    """
    predict    = xr.zeros_like(data.isel(variable=0).drop_vars('variable'))
    predict[:] = model.predict(data.values, quantiles='mean', aggregate_leaves_first=False)

    # Convert interval percentages to quantile bounds
    if Config.model.intervals:
        quantiles = []
        for percent in Config.model.intervals:
            lower = (1 - percent / 100) / 2
            upper = 1 - lower
            quant = [lower, upper]

            quantiles += quant

        # Fix any machine precision errors
        Config.model.quantiles = list(np.unique(np.round(quantiles, 3)))

    if Config.model.quantiles:
        quantiles = Config.model.quantiles
        Logger.debug(f'Predicting quantiles: {quantiles}')

        predicts = model.predict(data.values, quantiles=quantiles)
        Logger.debug('Finished predicting')

        # Set name so xr.merge is seamless
        predict.name = 'predict'

        arrays = [predict]
        for quantile, data in zip(quantiles, predicts.T):
            quant      = xr.full_like(predict, np.nan)
            quant[:]   = data
            quant.name = str(quantile)
            arrays.append(quant)

        save_objects(
            output    = output,
            kind      = kind,
            quantiles = xr.merge(arrays)
        )

    return predict


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
    # Verify there is an output directory, otherwise disable outputting
    if not output:
        if Config.output.path:
            output = Config.output.path
        else:
            Logger.warning(f'No output provided, disabling saving objects')
            for key in Config.output:
                Config.output[key] = False

    if output:
        mkdir(output)

    # Load the model from a pickle if provided
    if model is None:
        try:
            model = load_pkl(Config.input.model)
        except:
            Logger.exception(f'Failed to load model')
            return 1

    # Load data if not provided
    if data is None:
        data, target = Loader.load(split=True)

    # Stores information about this analysis
    stats = Sect()

    # Prepare the storage variables
    predict, bias, contributions = None, None, None
    # Quantile forests handle predictions tad differently
    if 'RandomForestQuantileRegressor' in str(model):
        predict = quantilePredict(
            model  = model,
            data   = data,
            output = output,
            kind   = kind
        )

    if Config.treeinterpreter is not False and 'Forest' in str(model):
        predict, bias, contributions = pbc(model, data)
    elif predict is None:
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
        if Config.output.scores:
            Logger.info(f'Saving scores to {output}/{kind}.scores.txt')
            with open(f'{output}/{kind}.scores.txt', 'w') as file:
                file.write('Scores:\n')
                file.write('\n'.join(scores))

    # Feature importances
    impout = None
    if Config.output.importance:
        impout = f'{output}/{kind}.importance.txt'
        Logger.info(f'Saving importances to {impout}')

    if 'Forest' in str(model):
        stats.mimportance = model_importance(model, data['variable'], output=impout)

    if Config.permutation_importance:
        try:
            stats.pimportance = perm_importance(model, data, target, output=impout)
        except:
            Logger.exception('Failed to generate permutation importance')

    # Create plots if enabled
    if Config.output.plots:
        if Config.plots.importance is not False:
            if stats.imports is not Null:
                plots.importance(
                    df   = stats.mimportance,
                    pdf  = stats.pimportance,
                    save = f'{output}/{kind}.importances.png'
                )

        if target is not None:
            if Config.plots.compare_target_predict is not False:
                plots.compare_target_predict(
                    target.unstack().mean('time').sortby('lon'),
                    predict.unstack().mean('time').sortby('lon'),
                    reindex = Config._reindex,
                    title   = f'{kind.capitalize()} Set Performance',
                    save    = f'{output}/{kind}.compare_target_predict.png'
                )

            if Config.plots.truth_vs_predicted is not False:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/Config.yaml',
                                            help     = 'Path to a Config.yaml file'
    )
    parser.add_argument('-p', '--patch',    nargs    = '?',
                                            metavar  = 'sect1 ... sectN',
                                            help     = 'Patch Sects together starting from sect1 to sectN'
    )

    args = parser.parse_args()

    init(args)

    folds = glob(f'{Config.output.path}/*/')
    Logger.debug(f'Folds discovered: {folds}')
    for fold in folds:
        Logger.info(f'Beginning analysis for fold: {fold}')
        model, data, target = load_from_run(fold, 'test', ['model', 'data', 'target'], load=True, stack=True)
        data = data.transpose()
        target = target.squeeze()

        if model is None:
            Logger.error(f'Missing model file')
            continue

        if data is None:
            Logger.error(f'Missing data file')
            continue

        if target is None:
            Logger.error(f'Missing target file')
            continue

        analyze(model, data, target, kind='test')
        Logger.debug(f'Finished fold: {fold}')
