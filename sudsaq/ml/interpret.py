"""
"""
import logging

import click

from mlky import Config

from sudsaq.data  import flatten
from sudsaq.ml    import treeinterpreter as ti
from sudsaq.utils import (
    load_from_run,
    save_objects
)

Logger = logging.getLogger(__file__)

@click.command()
@click.option('-p', '--path', help='Path to a SUDSAQ output directory that contains a model.pkl')
@click.option('-n', '--n_jobs', type=int, help='Sklearn n_jobs: -1 = all cores, int = N cores', default=-1)
@click.option('-l', '--log', type=click.Choice(['debug', 'info', 'warning', 'error'], case_sensitive=False), default='debug')
@click.option('-p', '--path', help='Path to a SUDSAQ output directory that contains a model.pkl')
@click.option('--predicts/--no-predicts', ' /-np', default=True, help='Enable/disable saving predicts')
@click.option('--biases/--no-biases', ' /-nb', default=True, help='Enable/disable saving biases')
@click.option('--contributions/--no-contributions', ' /-nc', default=True, help='Enable/disable saving contributions')
def main(path, n_jobs, log, predicts, biases, contributions):
    """
    Applies the TreeInterpreter algorithm against an existing SUDSAQ output
    """
    logging.basicConfig(
        level   = log.upper(),
        format  = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt = '%m-%d %H:%M'
    )

    Config.output = {'predict': predicts, 'bias': biases, 'contributions': contributions}

    Logger.info('Loading run into memory')

    model, data = load_from_run(path, 'test', ['model', 'data'])
    data.load()

    Logger.info('Flattening the data')

    flat = flatten(data).dropna('loc').transpose('loc', 'variable')
    pred = flat.isel(variable=0).drop('variable')
    bias = pred.copy()

    pred.name = 'predict'

    Logger.info(f'Predicting TreeInterpreter with X.shape = {flat.data.shape}')
    predicts, biases, contribs = ti.predict(model, flat.data, n_jobs=n_jobs)

    pred[:] = predicts.ravel()
    bias[:] = biases
    flat[:] = contribs

    save_objects(
        output        = path,
        kind          = 'test',
        predict       = pred,
        bias          = bias,
        contributions = flat
    )
    Logger.info('Finished')


if __name__ == '__main__':
    main()
