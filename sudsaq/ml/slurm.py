"""
SLURM job creator for SUDSAQ
"""
# Builtin
import argparse
import logging
import os
import pathlib
import sys

from datetime import datetime as dtt

# External
import numpy as np

from mlky import (
    Config,
    Sect
)
from mlky.utils import printTable

# Internal
import sudsaq

from sudsaq.utils import (
    load_pkl,
    save_pkl
)


logging.basicConfig(
    level    = logging.DEBUG,
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt  = '%m-%d %H:%M'
)

Logger = logging.getLogger('slurm.py')

#%%
SLURM = """\
#!/bin/bash
#SBATCH -J {name}
#SBATCH -A sudsaq
#SBATCH -o {logs}/oe/out.%a
#SBATCH -o {logs}/oe/err.%a
#SBATCH -c {cpu}
#SBATCH -p defq
#SBATCH -t 240:00:00
#SBATCH --array={array}

# Utility exports
export JOBLIB_TEMP_FOLDER=/tmp
export HDF5_USE_FILE_LOCKING=FALSE

# Use actual month numbers to index the array
month=('' # Offsets the months to their actual number
    jan feb mar
    apr may jun
    jul aug sep
    oct nov dec
)

# Activate the shared micromamba environment
. /cm/shared/apps/conda/etc/profile.d/conda.sh
conda activate /scratch/sudsaq/micromamba/envs/sudsaq

# Refresh MLIA_active_data
ls /projects/mlia-active-data/data_SUDSAQ/data

# Used by mlky.Config to generate the output directory at runtime
export MODELNAME={model}

python /scratch/sudsaq/suds-air-quality/sudsaq/ml/{script}.py -c {config} -p "{patch}<-${{month[$SLURM_ARRAY_TASK_ID]}}" {extra}
"""


def writeScript(file, script):
    """
    Helper function to write a script template
    """
    with open(str(file), 'w') as out:
        out.write(script)
    os.chmod(file, 0o775)

#%%
squeueTemplate = f"""\
#!/bin/bash

squeue -a -u {os.getlogin()}
"""
def squeue(logs):
    """
    Prints the squeue for the current user
    """
    writeScript(
        file   = logs/'squeue.sh',
        script = squeueTemplate
    )

launchTemplate = """\
#!/bin/bash

echo "Cleaning oe/*"
rm oe/*

echo "Launching job"
sbatch job.slurm
"""
def launch(logs):
    writeScript(
        file   = logs/'launch.sh',
        script = launchTemplate
    )

#%%
lsTemplate = """\
#!/bin/bash

{cmds}
"""
def ls(logs, patch, months):
    """
    Lists the model directories for a given run
    """
    ls = []
    for month in months:
        dir = Config.resetSects(f'{patch}<-{month}').output.path
        cmd = f'ls "{dir}"/**'
        ls.append('echo ""')
        ls.append(f"echo 'Section {month!r}: {cmd}'")
        ls.append(cmd)

    writeScript(
        file   = logs/'ls_run.sh',
        script = lsTemplate.format(cmds='\n'.join(ls))
    )

#%%
tail1Template = """\
#!/bin/bash

tail -n 1 {files}
"""
def tail1(logs, patch, months):
    """
    Tails 1 on each log file for a given run
    """
    files = []
    for month in months:
        log = Config.resetSects(f'{patch}<-{month}').log.file

        if log:
            files.append(f'"{log}"')

    if files:
        writeScript(
            file   = logs/'tail1.sh',
            script = tail1Template.format(files=' \\\n '.join(files))
        )

#%%
Months = np.array(['', # Offsets months to match their real numbers
    'jan', 'feb', 'mar',
    'apr', 'may', 'jun',
    'jul', 'aug', 'sep',
    'oct', 'nov', 'dec',
])
def parseArray(array):
    """
    """
    if '-' in array:
        step = 1
        start, stop = array.split('-')
        if ':' in stop:
            stop, step = stop.split(':')
        return list(range(int(start), int(stop)+1, int(step)))
    elif ',' in array:
        return [int(i) for i in array.split(',')]

#%%


def createJob(config, patch, logs, array, months, script, history, preview):
    """
    Generates a SLURM job for the SUDSAQ pipeline
    """
    # Preprocessing for the job template
    jid   = len(list(logs.glob('*'))) # Relative sudsaq job ID
    model = '.'.join(patch)           # Model name
    logs /= f'{jid}.{model}'          # Update logs to be the subdir for this run

    # Insert into the environ so script generation works correctly with mlky.replace reliance
    os.environ['MODELNAME'] = model

    # Extra flags to include for specific scripts
    if script == 'create':
        extra = '--restart'
    elif script == 'explain':
        extra = '--kind test'

    # Build the SLURM job shell script
    job = SLURM.format(
        name   = f'{jid}.{model}',
        logs   = logs,
        cpu    = 64,
        # mem    = 500,
        array  = array,
        model  = model,
        script = script,
        config = config,
        patch  = '<-'.join(patch),
        extra  = extra
    )

    # Scripts will just use the string form
    patch = '<-'.join(patch)

    if preview:
        history['launched'] = False
        Logger.info(f'Job to be submitted:\n"""\n{job}"""')
        return

    Logger.info('Preparing job for launch')

    # Make sure the logs directory exists
    (logs / 'oe').mkdir(mode=0o775, parents=True, exist_ok=True)

    # Create the utility scripts
    squeue(logs)
    launch(logs)
    ls(logs, patch, months)
    tail1(logs, patch, months)

    script = logs/'job.slurm'
    writeScript(
        file   = script,
        script = job
    )

    Logger.info(f'Launching job {logs}/job.slurm')
    history['slurmID']  = os.popen(f'sbatch {script}').read().strip().split()[-1]
    history['launched'] = True
    history['sudsaqID'] = jid,
    history['logs']     = logs

    Logger.info(f"SLURM ID is {history['slurmID']}, SUDSAQ ID is {history['sudsaqID']}")
    with open(logs/'info.txt', 'w') as output:
        printTable(history.items(), prepend='\n', print=output.write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            metavar  = '/path/to/config.yaml',
                                            default  = '/scratch/sudsaq/suds-air-quality/sudsaq/configs/definitions.yml',
                                            help     = 'Path to a config.yml file'
    )
    parser.add_argument('-p', '--patch',    nargs    = '?',
                                            metavar  = 'sect1 ... sectN',
                                            required = True,
                                            help     = 'Patch sections together starting from sect1 to sectN'
    )
    parser.add_argument('-m', '--months',   type     = str,
                                            default  = "1-12",
                                            help     = 'Months of SUDSAQ to execute, format as either a range "1-12" or list "1,3,5"'
    )
    parser.add_argument('-x', '--script',   choices  = ['create', 'explain'],
                                            default  = 'create',
                                            metavar  = 'script',
                                            help     = 'Which script to run'
    )
    parser.add_argument('-l', '--logs',     type     = str,
                                            metavar  = '/path/to/logs/directory/',
                                            default  = '/scratch/sudsaq/logs/',
                                            help     = 'Path to write SUDSAQ logs to; creates a subfolder for this job'
    )
    parser.add_argument('--preview',        action   = 'store_true',
                                            help     = 'Sets up the job and prints a preview without launching'
    )
    parser.add_argument('--history',        action   = 'store_true',
                                            help     = 'Prints the history of this script'
    )

    args = parser.parse_args()

    # Verify the logs directory is valid
    logs = pathlib.Path(args.logs).resolve()
    if not logs.exists():
        Logger.error(f'The logs directory must exist: {logs}')
        sys.exit(1)
    elif not logs.is_dir():
        Logger.error(f'The logs directory must be a directory: {logs}')
        sys.exit(4)

    # Either load or create the history dict
    history = {}
    hfile   = logs/'history.pkl'
    if hfile.exists():
        history = load_pkl(hfile)

    run = history[len(history)] = {
        'cmd' : 'python ' + ' '.join(sys.argv),
        'time': dtt.now()
    }

    # Just print the history
    if args.history:
        for id, run in history.items():
            Logger.info(f'History ID: {id}')
            align_print(run, prepend='  ', print=Logger.info)
            sys.exit(0)

    # Extra checks before creating the job
    Logger.info('Verifying arguments')

    # Make sure the config actually exists
    config = pathlib.Path(args.config).resolve()
    if not config.exists():
        Logger.error(f'Config file not found: {config}')
        sys.exit(2)

    # Convert patch to list form if needed
    if isinstance(args.patch, str):
        patch = args.patch.split('<-')

    # Retrieve the actual month names
    months = Months[parseArray(args.months)] # Months selected for this run

    # Check if these are in this config
    Config(config) # Load sections onto root

    for section in patch + list(months):
        if section not in Config:
            Logger.error(f'Section not in config: {section!r}')
            sys.exit(3)

    createJob(
        config  = config,
        patch   = patch,
        logs    = logs,
        array   = args.months,
        months  = months,
        script  = args.script,
        history = run,
        preview = args.preview
    )

    # Don't save when it's a preview
    if not args.preview:
        Logger.info('Saving history')
        save_pkl(history, hfile)
        Logger.info('Done')
