"""
"""
import argparse
import joblib
import logging
import numpy  as np
import xarray as xr

from mlky import (
    Config,
    Sect
)

from sklearn                 import ensemble as models
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold
)

try:
    from quantile_forest import RandomForestQuantileRegressor
except:
    RandomForestQuantileRegressor = None

from sudsaq.data       import load
from sudsaq.ml.analyze import analyze
from sudsaq.ml.explain import explain
from sudsaq.utils      import (
    align_print,
    init,
    save_objects,
    save_pkl
)


Logger = logging.getLogger('sudsaq/ml/create.py')


def prepare(kind, data, target, drop_features=False, align=False):
    """
    """
    # Make sure the data is loaded into memory
    Logger.debug(f'Loading {kind}ing data')
    data   = data.load()
    target = target.load()

    # Replace inf values if they exist
    Logger.debug('Replacing inf values if they exist')
    data   = data.where(np.isfinite(data), np.nan)
    target = target.where(np.isfinite(target), np.nan)

    # Detect if a given feature will drop everything
    counts = data.count('loc')
    feats  = data['variable'][counts == 0]
    if any(feats):
        Logger.error(f'{feats.size} features for this fold will cause all data to be dropped due to NaNs, see debug for more')
        for feat in feats.values:
            Logger.debug(f'  - {feat}')

        # Attempt to drop the problematic features so the fold can continue forwards
        # if drop_features:
        #     # TODO: Implement?

    # Always remove NaNs on the training set
    Logger.debug(f'Dropping NaNs from {kind}ing data')
    data   = data.dropna('loc')
    target = target.dropna('loc')

    if align:
        Logger.debug(f'Aligning {kind}ing data')
        data, target = xr.align(data, target, copy=False)

    Logger.debug(f'{kind} set stats:')
    Logger.debug(f'- Data   shape: {list(zip(data.dims, data.shape))}')
    Logger.debug(f'- Target shape: {list(zip(target.dims, target.shape))}')
    Logger.debug(f'Memory footprint in GB:')
    Logger.debug(f'- Data   = {data.nbytes / 2**30:.3f}')
    Logger.debug(f'- Target = {target.nbytes / 2**30:.3f}')

    if target.size == 0:
        Logger.error(f'{kind} target detected to be empty')
        return None, None
    if data.size == 0:
        Logger.error(f'{kind} data detected to be empty')
        return None, None

    return data, target


def fit(model, data, target, fold=None):
    """
    Fits a model and analyzes the performance

    Parameters
    ----------
    model: sklearn.ensemble.*
        An sklearn ensemble model. Tested well with RandomForestRegressor, all
        other models are experimental
    data: mlky.Sect
        Sect object containing the .train and .test splits for this X
        These splits should be xarray.Dataset
    target: mlky.Sect
        Sect object containing the .train and .test splits for this y
        These splits should be xarray.Dataset

    Notes
    -----
    The test set may not want to have NaNs
    The training set must have its NaNs dropped
    Loading is done in this function to ensure
    memory usage is kept down. Lazy load before
    this function.

    Cannot drop NaNs on target prior to kfold as
    X and y wouldn't align raising an exception
    """
    Logger.info('Preparing data for model training')

    # Prepare the training data
    data.train, target.train = prepare('train', data.train, target.train, align=True)

    if data.train is None or target.train is None:
        Logger.error('No train data available, skipping fold')
        return

    # Train the model as well as any Quantile Random Forests if given
    Logger.info('Training model')
    model.fit(data.train, target.train)

    # Create a subdirectory if kfold
    output = Config.output.path
    if fold:
        year = set(target.test.time.dt.year.values).pop()
        Logger.info(f'Testing year: {year}')
        if output:
            output += f'/{year}/'

    if Config.output.model:
        Logger.info(f'Saving model to {output}/model.pkl')
        save_pkl(model, f'{output}/model.pkl')

    # Run analyze on the train set
    if Config.analyze.train:
        Logger.info(f'Creating train set performance analysis')
        analyze(model, data.train, target.train, 'train', output)
    else:
        Logger.debug('Saving train objects')
        save_objects(
            output = output,
            kind   = 'train',
            data   = data.train,
            target = target.train
        )

    # Run analyze on the test set
    if Config.analyze.test:
        data.test, target.test = prepare('test', data.test, target.test, align=Config.align_test)

        if data.test is None or target.test is None:
            Logger.error('Test set analysis for this fold is unavailable')
            return

        Logger.debug('Saving test objects')
        save_objects(
            output  = output,
            kind    = 'test',
            data    = data.test,
            target  = target.test
        )

        Logger.info(f'Creating test set performance analysis')
        try:
            analyze(
                model  = model,
                data   = data.test,
                target = target.test,
                kind   = 'test',
                output = output
            )
        except:
            Logger.exception('Test analysis raised an exception')
            Critical.append(f'Test analysis failed for fold {fold}')

        # Run the explanation module if it's enabled
        try: # TODO: Remove the try/except, ideally module will handle exceptions itself so this is temporary
            if Config.explain:
                explain(
                    model  = model,
                    data   = data.test,
                    kind   = 'test',
                    output = output
                )
        except:
            Logger.exception('SHAP explanations failed:')


def hyperoptimize(data, target, model, kfold=None, groups=None):
    """
    Searches hyperparameter space to determine the best parameters
    to use for a given model and data input.

    Parameters
    ----------
    data:
    target:
    model:

    Returns
    -------
    lambda
        Lambda function that creates an instance of the provided model with the optimized parameters
    """
    Logger.info('Performing hyperparameter optimization (this may take awhile)')

    # Load all of the data, drop the nans, and align the time dimension
    data, target = xr.align(
        data.load().dropna('loc'),
        target.load().dropna('loc')
    )

    gscv = GridSearchCV(
        estimator   = model(),
        param_grid  = dict(Config.hyperoptimize.param_grid),
        cv          = kfold,
        error_score = 'raise',
        **Config.hyperoptimize.GridSearchCV
    )

    gscv.fit(data, target, groups=groups)

    Logger.info('Optimal parameter selection:')
    align_print(gscv.best_params_, enum=False, prepend='  ', print=Logger.info)

    # Create the predictor
    return lambda: model(**Config.model.params, **gscv.best_params_)


def create():
    """
    Creates and trains a desired model.
    """
    # Load the data
    data, target = load(split=True)

    ## Select a model to use per the config
    # Sklearn
    if Config.model.kind in dir(models):
        Logger.info(f'Selecting {Config.model.kind} as model')
        ensemble = getattr(models, Config.model.kind)

    # Zillow RandomForestQuantileRegressor
    elif Config.model.kind == 'RandomForestQuantileRegressor':
        if RandomForestQuantileRegressor is None:
            Logger.error('Please install https://github.com/zillow/quantile-forest before using RandomForestQuantileRegressor')
            return True

        ensemble = RandomForestQuantileRegressor

    # Error, not supported
    else:
        Logger.info(f'Invalid model selection: {Config.model.kind}')
        return 1

    # If KFold is enabled, set it up
    kfold  = None
    groups = None
    if Config.KFold:
        Logger.debug('Using KFold')
        kfold = KFold(**Config.KFold)

    # Split using grouped years
    elif Config.GroupKFold:
        Logger.debug('Using GroupKFold')
        kfold  = GroupKFold(n_splits=len(set(data.time.dt.year.values)))
        groups = target.time.dt.year.values

    if Config.hyperoptimize:
        model = hyperoptimize(data, target, ensemble, kfold=kfold, groups=groups)
    else:
        model = lambda: ensemble(**Config.model.params)

    # Start training and testing
    if kfold:
        for fold, (train, test) in enumerate(kfold.split(data, target, groups=groups)):
            input = Sect({
                'data': {
                    'train': data.isel(loc=train),
                    'test' : data.isel(loc=test)
                },
                'target': {
                    'train': target.isel(loc=train),
                    'test' : target.isel(loc=test)
                }
            })
            Logger.debug(f'fold_{fold}: Train years = {set(input.target.train.time.dt.year.values)}')
            Logger.debug(f'fold_{fold}:  Test years = {set(input.target.test.time.dt.year.values)}')

            # Recreate the model each fold
            fit(model(), input.data, input.target, fold=fold)

    # Not using kfold
    else:
        input = Sect({
            'data'  : {'train': data},
            'target': {'train': target}
        })

        # Load a different test set in, if available
        if Config.input.test:
            input.data.test, input.target.test = load(input=Config.input.test, split=True)
        else:
            Config.analyze.test = False

        fit(model(), input.data, input.target)

    # Completed successfully
    return True


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
    parser.add_argument('--restart',        action   = 'store_true',
                                            help     = 'Will auto restart the run until the state returns True'
    )

    args = parser.parse_args()

    init(args)

    Critical = []
    state    = False
    loop     = 1
    while args.restart or loop == 1:
        try:
            # with joblib.parallel_backend('dask'):
            state = create()
        except Exception:
            Logger.exception('Caught an exception during runtime')
        finally:
            if state is True:
                if Critical:
                    Logger.info('Critical errors:')
                    for msg in Critical:
                        Logger.error(f'- {msg}')
                    Logger.info('Finished gracefully but with critical errors, see above')
                else:
                    Logger.info('Finished successfully')
                break
            else:
                Logger.error(f'Failed to complete with status code: {state}')
                loop += 1
                if loop < 10:
                    Logger.error(f'Restarting, attempt {loop}')
                else:
                    Logger.error('10 attempts have failed, exiting')
                    break
