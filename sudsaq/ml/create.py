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

from sudsaq import (
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

    data.train, target.train = prepare('train', data.train, target.train, align=True)

    Logger.info('Training model')
    model.fit(data.train, target.train)

    # Create a subdirectory if kfold
    output = config.output.path
    if test:
        year = set(target.test.time.dt.year.values).pop()
        Logger.info(f'Testing year: {year}')
        if output:
            output += f'/{year}/'

    if config.output.model:
        Logger.info(f'Saving model to {output}/model.pkl')
        save_pkl(model, f'{output}/model.pkl')

    if config.train_performance:
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

    if test:
        data.test, target.test = prepare('test', data.test, target.test, align=config.align_test)

        if data.test is None or target.test is None:
            Logger.error('Test set analysis for this fold was cancelled')
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
            Critical.append(f'Test analysis failed for fold {i}')

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
    config = Config()

    # Load all of the data, drop the nans, and align the time dimension
    data, target = xr.align(
        data.load().dropna('loc'),
        target.load().dropna('loc')
    )

    gscv = GridSearchCV(
        estimator   = model(),
        param_grid  = dict(config.hyperoptimize.param_grid),
        cv          = kfold,
        error_score = 'raise',
        **config.hyperoptimize.GridSearchCV
    )

    gscv.fit(data, target, groups=groups)

    Logger.info('Optimal parameter selection:')
    align_print(gscv.best_params_, enum=False, prepend='  ', print=Logger.info)

    # Create the predictor
    return lambda: model(**config.model.params, **gscv.best_params_)

def create():
    """
    Creates and trains a desired model.
    """
    # Load config and data
    config       = Config()
    data, target = load(config, split=True)

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
        model = hyperoptimize(data, target, ensemble, kfold=kfold, groups=groups)
    else:
        model = lambda: ensemble(**config.model.params)

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
    parser.add_argument('-i', '--inherit',  nargs    = '?',
                                            metavar  = 'sect1 sect2',
                                            help     = 'Order of keys to apply inheritance where rightmost takes precedence over left'
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
