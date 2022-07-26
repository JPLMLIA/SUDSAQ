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
from sudsaq.utils      import (
    align_print,
    init
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
    i: int, default = None
        The fold iteration, creates a folder under output
    test: bool, default = None

    """
    # Retrieve the config object
    config = Config()

    # Create a subdirectory if kfold
    output = config.output.path
    if test:
        year = set(target.test.time.dt.year.values).pop()
        Logger.info(f'Training year: {year}')
        if output:
            output = f'{output}/{year}/'

    Logger.info('Training model')
    model.fit(data.train, target.train)

    if config.train_performance:
        Logger.info(f'Creating train set performance analysis')
        analyze(model, data.train, target.train, 'train', output)

    if test:
        Logger.info(f'Creating test set performance analysis')
        analyze(model, data.test, target.test, 'test', output)

def create():
    """
    Creates and trains a desired model.
    """
    # Load config and data
    config       = Config()
    data, target = load(config, split=True, lazy=False)

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

            Logger.debug(f'fold_{fold} Is Finite')
            Logger.debug('Data   :')
            Logger.debug(f' train: {np.isfinite(input.data.train).all().values}')
            Logger.debug(f'  test: {np.isfinite(input.data.test).all().values}')
            Logger.debug('Target :')
            Logger.debug(f' train: {np.isfinite(input.target.train).all().values}')
            Logger.debug(f'  test: {np.isfinite(input.target.test).all().values}')

            Logger.debug(f'fold_{fold} Has NaN')
            Logger.debug('Data   :')
            Logger.debug(f' train: {np.isnan(input.data.train).any().values}')
            Logger.debug(f'  test: {np.isnan(input.data.test).any().values}')
            Logger.debug('Target :')
            Logger.debug(f' train: {np.isnan(input.target.train).any().values}')
            Logger.debug(f'  test: {np.isnan(input.target.test).any().values}')

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
