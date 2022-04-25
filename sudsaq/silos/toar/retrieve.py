"""
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import requests

from datetime import datetime
from glob     import glob
from tqdm     import tqdm

from sudsaq.config import Config
from sudsaq.utils  import init


def download(parameter):
    """
    Performs GET operations to the TOAR2 API for the given parameter, tweaked by config

    Parameters
    ----------
    parameter: str
        Name of the parameter to query

    Returns
    -------
    errors: list
        List of errors incurred while downloading

    Notes
    -----
    The configuration file can control the parameters of the API call

    Config
    ------
    sampling: str
    metrics: list
    daterange:
        start: datetime
        end: datetime
    parameters: list
    networks: list
    stations: list
    output: str
    overwrite: bool
    """
    # Retrieve the config object
    config = Config()

    # Lookup the records for this parameter and only select from the networks and stations desired
    resp    = requests.get(f'{config.url}/series/?parameter_name={parameter}&as_dict=True')
    records = resp.json()
    records = [record for record in records if all([record['network_name'] in config.networks, record['station_id'] in config.stations])]
    errors  = []
    for record in tqdm(records, desc=parameter, position=config.parameters.index(parameter)):
        id, network, station, param = record.values()

        if config.daterange:
            resp = requests.get(f'{config.url}/stats/?sampling={config.sampling}&statistics={config.metrics}&daterange={config.daterange}&id={id}')
        else:
            resp = requests.get(f'{config.url}/stats/?sampling={config.sampling}&statistics={config.metrics}&id={id}')

        if resp:
            data = resp.json()
            if len(data) <= 2:
                errors.append(data)
            else:
                # Check if directories already exist, create if not
                # mode 771 = rwx,rwx,--x; self,group,others
                if not os.path.exists(f'{config.output}/{network}'):
                    os.mkdir(f'{config.output}/{network}', mode=0o771)

                if not os.path.exists(f'{config.output}/{network}/{station}/'):
                    os.mkdir(f'{config.output}/{network}/{station}/', mode=0o771)

                # Create the output directory for this parameter
                outdir = f'{config.output}/{network}/{station}/{parameter}/'
                if not os.path.exists(outdir):
                    os.mkdir(outdir, mode=0o771)

                output = f'{outdir}/{id}.json'
                if not config.overwrite:
                    if output in glob(outdir):
                        Logger.debug(f'File already exists: {output}')
                        continue

                with open(output, 'w') as file:
                    json.dump(data, file)

    return parameter, errors

def retrieve():
    """
    """
    # Retrieve the config object
    config = Config()

    # Retrieve all parameters if not provided or all requested
    if config.parameters in [None, 'all']:
        Logger.info('Retrieving all parameters')
        resp = requests.get(f'{config.url}/parameters/')
        if resp:
            config.parameters = resp.json()
        else:
            print(f'Failed to retrieve parameters: {resp}')
    Logger.info(f'Number of parameters: {len(config.parameters)}')

    # Retrieve all networks if not provided or all requested
    if config.networks in [None, 'all']:
        Logger.info('Retrieving all networks')
        resp = requests.get(f'{config.url}/networks/')
        if resp:
            config.networks = resp.json()
        else:
            print(f'Failed to retrieve stations: {resp}')
    Logger.info(f'Number of networks: {len(config.networks)}')

    # Format networks list to be RESTful
    config.networks = ','.join(config.networks)

    # Retrieve all stations if not provided or all requested
    if config.stations in [None, 'all']:
        Logger.info('Retrieving all stations')
        resp = requests.get(f'{config.url}/search/?columns=station_id')
        if resp:
            config.stations = [station for station, in resp.json()]
        else:
            print(f'Failed to retrieve stations: {resp}')
    Logger.info(f'Number of stations: {len(config.stations)}')

    # Format metrics to be RESTful
    Logger.info(f'Number of stations: {len(config.metrics)}')
    config.metrics = ','.join(config.metrics)

    # Format the date if given
    if config.daterange:
        Logger.info(f'Enabling daterange for query: {config.daterange.start} - {config.daterange.end}')
        config.daterange = f'[{requests.utils.quote(config.daterange.start)},{requests.utils.quote(config.daterange.end)}]'

    Logger.info('Starting downloads')
    all_errors = {}
    bar = tqdm(total=len(config.parameters), desc='Parameters downloaded', position=0)
    with mp.Pool(processes=config.processes or 1) as pool:
        for parameter, errors in pool.imap(download, config.parameters):
            all_errors[parameter] = errors
            bar.update()

    # Write the errors out
    Logger.info(f'Saving errors encountered to text file')
    if all_errors:
        if not os.path.exists(f'{config.output}/errors'):
            os.mkdir(f'{config.output}/errors', mode=0o771)

        ts = datetime.now().strftime('%m_%d_%Y.%H_%M')
        output = f'{config.output}/errors/{ts}.txt'
        Logger.debug(f'Writing to: {output}')
        with open(output, 'w') as file:
            file.write(f'Errors collected during run:\n')
            for parameter, errors in all_errors.items():
                file.write(f'{parameter}:\n')
                for i, error in enumerate(errors):
                    file.write(f'- {i:03}: {error}\n')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'retrieve',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    init(parser.parse_args())
    Logger = logging.getLogger('sudsaq/silos/toar/retrieve.py')

    state = False
    try:
        state = retrieve()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
