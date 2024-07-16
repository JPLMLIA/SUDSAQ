# Builtin
import logging
import os
import sys

from pathlib import Path

# External
import click
import mlky

from mlky import Config as C


Logger = logging.getLogger('sudsaq/cli')


def initConfig(config, patch, override, printconfig=False, printonly=False, print=print):
    """
    Initializes the mlky Config object
    """
    C(config, _patch=patch, _override=override)

    # Print configuration to terminal
    if printconfig or printonly:
        print(f'Config({config!r}, _patch={patch!r})')
        print('-'*100)
        print(C.toYaml(comments=None, listStyle='short', header=False))
        print('-'*100)

        if printonly:
            sys.exit()


def initLogging():
    """
    Initializes the logging module per the config
    """
    # Logging handlers
    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)

    if (level := C.log.terminal):
        sh.setLevel(level)

    handlers.append(sh)

    if (file := C.log.file):
        if C.log.mode == 'write' and os.path.exists(file):
            os.remove(C.log.file)

        # Create the directory path
        Path(file).parent.mkdir(parents=True, exist_ok=True)

        # Add the file logging
        fh = logging.FileHandler(file)
        fh.setLevel(C.log.level.upper() or logging.DEBUG)

        handlers.append(fh)

    logging.basicConfig(
        level    = C.log.get('level', 'DEBUG').upper(),
        format   = C.log.get('format', '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'),
        datefmt  = C.log.get('format', '%m-%d %H:%M'),
        handlers = handlers,
    )


@click.group(name='sudsaq')
def cli():
    """\
    TimeFED is a machine learning system for Time series Forecasting, Evaluation and Deployment.
    """
    ...



@cli.command(name='create', context_settings={'show_default': True})
@mlky.cli.config
@mlky.cli.patch
@mlky.cli.override
@click.option("-pc", "--printConfig", help="Prints the configuration to terminal and continues", is_flag=True)
@click.option("-po", "--printOnly", help="Prints the configuration to terminal and exits", is_flag=True)
def create(**kwargs):
    """\
    Create models using the SUDSAQ pipeline
    """
    initConfig(**kwargs, print=click.echo)
    initLogging()

    from sudsaq.ml.create import create

    create()


@cli.command(name='explain', context_settings={'show_default': True})
@click.option('-k', '--kind', default='test', type=click.Choice(['test', 'train']), help='Kind of data to explain')
@mlky.cli.config
@mlky.cli.patch
@mlky.cli.override
@click.option("-pc", "--printConfig", help="Prints the configuration to terminal and continues", is_flag=True)
@click.option("-po", "--printOnly", help="Prints the configuration to terminal and exits", is_flag=True)
def explain(kind, **kwargs):
    """\
    Explain SUDSAQ models using SHAP
    """
    initConfig(**kwargs, print=click.echo)
    initLogging()

    from sudsaq.ml.explain import explain

    explain()


# Add mlky as subcommands
mlky.cli.setDefaults(patch='default')
cli.add_command(mlky.cli.commands)


if __name__ == '__main__':
    cli()
