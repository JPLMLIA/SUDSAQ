"""
"""
import argparse
import logging
import numpy  as np
import xarray as xr

from sklearn                 import ensemble as models
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold
)

from sudsaq.config     import (
    Config,
    Section
)
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

def fit(model, data, target, i=None, test=True):
    """
    Fits a model and analyzes the performance

    Parameters
    ----------
    model:
    data:
    target:
    test: bool, default = None

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
    # Retrieve the config object
    config = Config()

    Logger.info('Preparing data for model training')

    # Make sure the data is loaded into memory
    Logger.debug('Loading training data')
    data.train   = data.train.load()
    target.train = target.train.load()

    # Always remove NaNs on the training set
    Logger.debug('Dropping NaNs from training data')
    data.train   = data.train.dropna('loc')
    target.train = target.train.dropna('loc')

    Logger.debug('Aligning training data')
    data.train, target.train = xr.align(data.train, target.train, copy=False)

    if target.train.size == 0:
        Logger.error('Train target detected to be empty, skipping this fold')
        return
    if data.train.size == 0:
        Logger.error('Train data detected to be empty, skipping this fold')
        return

    # Create a subdirectory if kfold
    output = config.output.path
    if test:
        year = set(target.test.time.dt.year.values).pop()
        Logger.info(f'Testing year: {year}')
        if output:
            output = f'{output}/{year}/'

    Logger.info('Training model')
    model.fit(data.train, target.train)

    if config.output.model:
        Logger.info(f'Saving model to {output}/model.pkl')
        save_pkl(model, f'{output}/model.pkl')

    if config.train_performance:
        Logger.info(f'Creating train set performance analysis')
        analyze(model, data.train, target.train, 'train', output)
    else:
        save_objects(
            output = output,
            kind   = 'train',
            data   = data.train,
            target = target.train
        )

    if test:
        Logger.debug('Loading test data')
        target.test = target.test.load()
        data.test   = data.test.load()

        Logger.debug('Dropping NaNs in test data')
        # Target and data drop NaNs separately for prediction, will be aligned afterwards
        data.test   = data.test.dropna('loc')
        target.test = target.test.dropna('loc')

        if config.align_test:
            Logger.debug('Aligning test data')
            data.test, target.test = xr.align(data.test, target.test, copy=False)

        if target.test.size == 0:
            Logger.warning('Test target detected to be entirely NaN, cancelling test analysis for this fold')
            return

        if data.test.size == 0:
            Logger.warning('Test data detected to be entirely NaN, cancelling test analysis for this fold')
            return

        Logger.info(f'Creating test set performance analysis')
        analyze(
            model  = model,
            data   = data.test,
            target = target.test,
            kind   = 'test',
            output = output
        )

        # Run the explanation module if it's enabled
        try: # TODO: Remove the try/except, ideally module will handle exceptions itself so this is temporary
            if config.explain:
                explain(
                    model  = model,
                    data   = data.test,
                    kind   = 'test',
                    output = output
                )
        except:
            Logger.exception('SHAP explanations failed:')

def create():
    """
    Creates and trains a desired model.
    """
    # Load config and data
    config       = Config()
    data, target = load(config, split=True, lazy=True)

    if config.model.kind in dir(models):
        Logger.info(f'Selecting {config.model.kind} as model')
        ensemble = getattr(models, config.model.kind)
    else:
        Logger.info(f'Invalid model selection: {config.model.kind}')
        return 1

    # If KFold is enabled, set it up
    kfold  = None
    groups = None
    if config.KFold:
        Logger.debug('Using KFold')
        kfold  = KFold(**config.KFold)
    # Split using grouped years
    elif config.GroupKFold:
        Logger.debug('Using GroupKFold')
        kfold  = GroupKFold(n_splits=len(set(data.time.dt.year.values)))
        groups = target.time.dt.year.values

    if config.hyperoptimize:
        Logger.info('Performing hyperparameter optimization')

        gscv = GridSearchCV(
            ensemble(),
            config.hyperoptimize.params._data,
            cv          = kfold,
            error_score = 'raise',
            **config.hyperoptimize.GridSearchCV
        )
        gscv.fit(data, target)

        Logger.info('Optimal parameter selection:')
        align_print(gscv.best_params_, enum=False, prepend='  ', print=Logger.info)

        # Create the predictor
        model = lambda: ensemble(**config.model_params, **gscv.best_params_)
    else:
        model = lambda: ensemble(**config.model_params)

    if kfold:
        for fold, (train, test) in enumerate(kfold.split(data, target, groups=groups)):
            input = Section('', {
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
            fit(model(), input.data, input.target, i=fold)
    else:
        input = Section({
            'data'  : {'train': data},
            'target': {'train': target}
        })
        fit(model(), input.data, input.target, test=False)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'create',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())

    state = False
    try:
        state = create()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
