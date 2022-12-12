"""
"""
import argparse
import os
import pathlib
import pickle
import sys

from datetime import datetime as dtt

import sudsaq

from sudsaq.config import Config
from sudsaq.utils  import (
    align_print,
    load_pkl,
    save_pkl
)

PBS = """\
#!/bin/bash
#PBS -N {user}{id}-sudsaq
#PBS -q array-sn
#PBS -l select=1:ncpus=48:mem=366gb
#PBS -l walltime=240:00:00
#PBS -e {logs}/$TASK_ID.error.txt
#PBS -o {logs}/$TASK_ID.output.txt
#PBS -v summary=true
#PBS -J {range}
#PBS -W group_list=mlia-active-data

# Activate the Conda environment
. /cm/shared/apps/conda/etc/profile.d/conda.sh
conda activate {env}

export JOBLIB_TEMP_FOLDER=$TMPDIR
export HDF5_USE_FILE_LOCKING=FALSE

SECTIONS=(\
{sections}
)

python {repo}/ml/create.py -c {config} -s ${_sects} --restart
"""

def create_job(file, sections, logs, preview=False, history={}):
    """
    """
    id   = len(list(logs.glob('job_*')))
    logs = f'{logs}/job_{id}'
    job  = PBS.format(
        user     = os.getlogin()[0], # Only take the first character for privacy
        id       = id,
        logs     = logs,
        range    = f'0-{len(sections)}',
        env      = os.environ['CONDA_DEFAULT_ENV'],
        sections = ''.join([f'\n\t"{sect}"' for sect in sections]),
        repo     = sudsaq.__path__[0],
        config   = file,
        _sects   = '{SECTIONS[$PBS_ARRAY_INDEX]}'
    )

    if preview:
        history['launched'] = False
        print(f'Job to be submitted:\n\n{job}\n')
    else:
        print('Preparing job for launch')
        os.mkdir(logs)

        with open(f'{logs}/job.pbs', 'w') as output:
            output.write(job)

        print(f'Launching job {logs}/job.pbs')
        print(os.system(f'qsub {logs}/job.pbs'))

        history['launched'] = True
        history['id']       = id,
        history['logs']     = logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--sections', type     = str,
                                            nargs    = '+',
                                            metavar  = '[sections, ...]',
                                            help     = 'Sections of the config to use for independent runs'
    )
    parser.add_argument('-l', '--logs',     type     = str,
                                            metavar  = '/path/to/logs/directory/',
                                            default  = './logs',
                                            help     = 'Path to write PBS logs to; creates a subfolder for this job'
    )
    parser.add_argument('-p', '--preview',  action   = 'store_true',
                                            help     = 'Sets up the job and prints a preview without launching'
    )
    parser.add_argument('--history',        action   = 'store_true',
                                            help     = 'Prints the history of this script'
    )

    args = parser.parse_args()

    logs = pathlib.Path(args.logs).resolve()
    if not logs.exists():
        print(f'Error: The logs directory must exist: {logs}')
        sys.exit(1)
    elif not logs.is_dir():
        print(f'Error: The logs directory must be a directory: {logs}')
        sys.exit(4)

    hfile   = f'{logs}/history.pkl'
    history = {}
    if os.path.exists(hfile):
        history = load_pkl(hfile)

    run = history[len(history)] = {
        'cmd' : ' '.join(sys.argv),
        'time': dtt.now()
    }

    if args.history:
        for id, run in history.items():
            print(f'Run ID: {id}')
            align_print(run, prepend='  ')
    else:
        file = pathlib.Path(args.config).resolve()
        if not file.exists():
            print(f'Error: Config file not found: {file}')
            sys.exit(2)

        for section in args.sections:
            try:
                Config(file, section)
            except:
                print(f'Section {section!r} encountered errors')
                sys.exit(3)

        create_job(file, args.sections, logs, preview=args.preview, history=run)

        print('Saving history')
        save_pkl(history, hfile)
        print('Done')
