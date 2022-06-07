"""
"""
import argparse
import logging
import tarfile
import xarray as xr

from glob import glob
from tqdm import tqdm

from sudsaq.config import Config
from sudsaq.utils  import init


def individual_folders():
    """
    """
    config = Config()

    for variable in tqdm(config.variables, desc='Processing Variables'):
        ds = xr.open_mfdataset(f'{config.path}/{variable}/{config.regex}', engine='scipy')

        if variable in config.rename:
            ds = ds.rename({config.rename[variable].var: config.rename[variable].name})

        ds.to_netcdf(f'{config.output}{variable}.nc', engine='scipy')
        del ds

def extract_tar(file):
    """
    """
    dss = []
    with tarfile.open(file, 'r:gz') as tar:
        for member in tqdm(tar.getmembers(), desc='Reading Members', position=0):
            extract = tar.extractfile(member)

            # Sometimes extractfile may return None (eg. not a file or link, ie. a directory)
            if extract:
                data = xr.open_dataset(extract, engine='scipy')

                # Check for generic variable name, replace it if so
                if 'var' in data:
                    variable = '/'.join(member.name[:-3].split('_')[1:])
                    data     = data.rename({'var': variable})

                dss.append(data)

    ds = xr.merge(dss)
    return ds

def process():
    """
    """
    config = Config()

    if config.individual_folders:
        Logger.info('Processing individual folders')
        individual_folders()
    else:
        # Retrieve only the tar.gz files and extract
        # merge = []
        tars = glob(f'{config.input}/*.tar.gz')
        Logger.debug(f'Reading {config.input}/*.tar.gz: found {len(tars)} files')

        for tar in tqdm(tars, desc='Extracting Tars', position=1):
            file = tar.split('/')[-1][:-7]
            out  = f'{config.output}/{file}.nc'

            if not glob(out) or config.regenerate:
                try:
                    ds = extract_tar(tar)
                    ds.to_netcdf(out, engine='scipy')
                    del ds
                except:
                    Logger.exception(f'Failed on file {tar}')
            # merge.append(ds)

        # Now merge these together
        # ds = xr.merge(merge)
        #
        # del merge
        #
        # Logger.info('Saving output')
        # ds.to_netcdf(config.output, engine='scipy')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'process',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())
    Logger = logging.getLogger('sudsaq/silos/momo/process.py')

    state = False
    try:
        state = process()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
