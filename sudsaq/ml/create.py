"""
"""
import argparse
import logging
import xarray as xr

from sklearn                 import ensemble as models
from sklearn.model_selection import (
    GridSearchCV,
    KFold
)

from sudsaq.config     import Config
from sudsaq.data       import load
from sudsaq.ml         import plots
from sudsaq.ml.analyze import analyze
from sudsaq.utils      import (
    align_print,
    init,
    save_pkl
)

Logger = logging.getLogger('sudsaq/ml/create.py')

def create():
    """
    Creates and trains a desired model.
    """
    # Load config and data
    config       = Config()
    data, target = load(config, split=True)

    if config.model.kind in dir(models):
        Logger.info(f'Selecting {config.model.kind} as model')
        model = getattr(models, config.model.kind)
    else:
        Logger.info(f'Invalid model selection: {config.model.kind}')
        return 1

    if config.hyperoptimize:
        Logger.info('Performing hyperparameter optimization')

        kfold = None
        if config.hyperoptimize.kfold:
            kfold = KFold(**config.hyperoptimize.kfold)

        gscv = GridSearchCV(
            model(),
            config.hyperoptimize.params._data,
            cv          = kfold,
            error_score = 'raise',
            **config.hyperoptimize.GridSearchCV
        )
        gscv.fit(data, target)

        Logger.info('Optimal parameter selection:')
        align_print(gscv.best_params_, enum=False, prepend=' '*4, print=Logger.info)

        # Create the predictor
        model = model(**config.model_params, **gscv.best_params_)
    else:
        model = model(**config.model_params)

    Logger.info('Training model')
    model.fit(data, target)

    if config.output.model:
        Logger.info(f'Saving model to {config.output.model}')
        save_pkl(config.output.model, model)

    if config.train_performance:
        Logger.info('Creating training performance analysis')

        predict = analyze(model, data, target).predict
        plots.compare_target_predict(
            target.unstack().mean('time').sortby('lon'),
            predict.unstack().mean('time').sortby('lon'),
            'Training Set Performance'
        )

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
