"""
"""
import logging

import click

from sudsaq.data  import flatten
from sudsaq.ml    import treeinterpreter as ti
from sudsaq.utils import (
    load_from_run,
    save_objects
)

Logger = logging.getLogger(__file__)

@click.command()
@click.option('-p', '--path', help='Path to a SUDSAQ output directory that contains a model.pkl')
def main(path):
    """
    Applies the TreeInterpreter algorithm against an existing SUDSAQ output
    """
    Logger.info('Loading run into memory')

    model, data = load_from_run(path, 'test', ['model', 'data'])
    data.load()

    Logger.info('Flattening data')
    data = flatten(data.dropna('loc')).data.T

    predicts, biases, contribs = ti.predict(model, X)

    save_objects(
        output  = path,
        kind    = 'test',
        bias    = biases,
        contributions = contribs
    )
    Logger.info('Finished')


if __name__ == '__main__':
    main()
