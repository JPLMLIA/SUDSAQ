"""
"""
import argparse
import logging
import os
import pathlib
import pickle
import sys

from datetime import datetime as dtt

from mlky import (
    Config,
    Section,
    replace
)

import sudsaq

from sudsaq.utils  import (
    align_print,
    load_pkl,
    save_pkl
)

logging.basicConfig(
    level    = logging.DEBUG,
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt  = '%m-%d %H:%M'
)

Logger = logging.getLogger('qsub.py')

Resources = Section('Resources', {
    'user': {
        'cpu': 384,
        'mem': 3000, # GB
        'walltime': 240 # Hours
    },
    'node': {
        'cpu': 48,
        'mem': 366 # GB
    }
})

PBS = """\
#!/bin/bash
#PBS -N {user}{id}-sudsaq
#PBS -q array-sn
#PBS -l select=1:ncpus={cpu}:mem={mem}gb
#PBS -l walltime=240:00:00
#PBS -e {logs}/job_{id}/
#PBS -o {logs}/job_{id}/
#PBS -v summary=true
#PBS -J {range}
#PBS -W group_list=mlia-active-data

# Activate the Conda environment
. /cm/shared/apps/conda/etc/profile.d/conda.sh
conda activate {env}

export JOBLIB_TEMP_FOLDER=$TMPDIR
export HDF5_USE_FILE_LOCKING=FALSE

ln -s {logs}/job_{id} {logs}/running/job_{id}

SECTIONS=(\
{sections}
)

export ID="{inherit}"

python {repo}/ml/{script}.py -c {config} -i "{inherit}<-${_sects}" {extra}

rm {logs}/running/job_{id}
"""

def qstat(logs, user):
    """
    """
    template = """\
#!/bin/bash

qstat -as @gattaca-edge -u {user}
    """
    with open(f'{logs}/qstat.sh', 'w') as output:
        output.write(template.format(user=user))
    os.chmod(f'{logs}/qstat.sh', 0o775)

def ls(logs, file, inherit, sections):
    """
    """
    template = """\
#!/bin/bash

{cmds}
    """
    ls = []
    for section in sections:
        dir = Config(file, f'{inherit}<-{section}').output.path
        cmd = f'ls {dir}/**'
        ls.append('echo ""')
        ls.append(f'echo "Section {section}: {cmd}"')
        ls.append(cmd)

    with open(f'{logs}/ls_run.sh', 'w') as output:
        output.write(template.format(cmds='\n'.join(ls)))
    os.chmod(f'{logs}/ls_run.sh', 0o775)

def tail1(logs, file, inherit, sections):
    """
    """
    template = """\
#!/bin/bash

tail -n 1 {files}
    """
    files = []
    for section in sections:
        log = Config(file, f'{inherit}<-{section}').log.file
        if log:
            files.append(log)

    if files:
        with open(f'{logs}/tail1.sh', 'w') as output:
            output.write(template.format(files=' '.join(files)))
        os.chmod(f'{logs}/tail1.sh', 0o775)

def create_job(file, inherit, sections, script, logs, n=1, preview=False, history={}):
    """
    """
    id   = len(list(logs.glob('job_*')))
    user = os.getlogin()

    if script == 'create':
        extra = '--restart'
    elif script == 'explain':
        extra = '--kind test'

    job  = PBS.format(
        user     = user[0], # Only take the first character for privacy
        id       = id,
        cpu      = min(Resources.node.cpu, int(Resources.user.cpu / n)),
        mem      = min(Resources.node.mem, int(Resources.user.mem / n)),
        logs     = logs,
        range    = f'0-{len(sections)-1}',
        env      = os.environ['CONDA_DEFAULT_ENV'],
        sections = ''.join([f'\n  "{sect}"' for sect in sections]),
        repo     = sudsaq.__path__[0],
        script   = script,
        extra    = extra,
        hash     = replace('${!hash}'),
        config   = file,
        inherit  = inherit,
        _sects   = '{SECTIONS[$PBS_ARRAY_INDEX]}'
    )
    logs = f'{logs}/job_{id}'

    if preview:
        history['launched'] = False
        Logger.info(f'Job to be submitted:\n"""\n{job}"""')
    else:
        Logger.info('Preparing job for launch')
        os.mkdir(logs)

        with open(f'{logs}/job.pbs', 'w') as output:
            output.write(job)

        Logger.info('Creating utility scripts for this job')
        qstat(logs, user)
        ls(logs, file, inherit, sections)
        tail1(logs, file, inherit, sections)

        Logger.info(f'Launching job {logs}/job.pbs')
        history['job_id']   = os.popen(f'qsub {logs}/job.pbs').read().replace('\n', '')
        history['launched'] = True
        history['run_id']   = id,
        history['logs']     = logs
        Logger.info(f"PBS ID is {history['job_id']}")

        with open(f'{logs}/info.txt', 'w') as output:
            align_print(history, prepend='\n', print=output.write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-i', '--inherit',  nargs    = '?',
                                            metavar  = 'sect1 sect2',
                                            help     = 'Order of keys to apply inheritance where rightmost takes precedence over left'
    )
    parser.add_argument('-s', '--sections', type     = str,
                                            nargs    = '+',
                                            metavar  = 'section',
                                            help     = 'Sections of the config to use for independent runs'
    )
    parser.add_argument('-x', '--script',   choices  = ['create', 'explain'],
                                            default  = 'create',
                                            metavar  = 'script',
                                            help     = 'Which script to run'
    )
    parser.add_argument('-l', '--logs',     type     = str,
                                            metavar  = '/path/to/logs/directory/',
                                            default  = './logs',
                                            help     = 'Path to write PBS logs to; creates a subfolder for this job'
    )
    parser.add_argument('-n', '--n_jobs',   type     = int,
                                            metavar  = 'int',
                                            default  = 1,
                                            help     = 'Minimum number of jobs to run concurrently. This will divide the user resource limits to determine max resources per node'
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
        Logger.error(f'The logs directory must exist: {logs}')
        sys.exit(1)
    elif not logs.is_dir():
        Logger.error(f'The logs directory must be a directory: {logs}')
        sys.exit(4)

    if not os.path.exists(f'{logs}/running'):
        os.mkdir(f'{logs}/running')

    hfile   = f'{logs}/history.pkl'
    history = {}
    if os.path.exists(hfile):
        history = load_pkl(hfile)

    run = history[len(history)] = {
        'cmd' : 'python ' + ' '.join(sys.argv),
        'time': dtt.now()
    }

    if args.history:
        for id, run in history.items():
            Logger.info(f'History ID: {id}')
            align_print(run, prepend='  ', print=Logger.info)
    else:
        Logger.info('Verifying arguments')
        file = str(pathlib.Path(args.config).resolve())
        if not os.path.exists(file):
            Logger.error(f'Config file not found: {file}')
            sys.exit(2)

        if isinstance(args.inherit, str):
            inherit = args.inherit
        elif len(args.inherit) == 1:
            inherit, = args.inherit
        else:
            inherit = '<-'.join(args.inherit)

        os.environ['ID'] = inherit

        for section in args.sections:
            try:
                Config(file, f'{inherit}<-{section}')
            except:
                Logger.exception(f'Failed to load {inherit}<-{section}')
                sys.exit(3)

        Logger.info('Creating job')
        create_job(
            file     = file,
            inherit  = inherit,
            sections = args.sections,
            script   = args.script,
            logs     = logs,
            n        = args.n_jobs,
            preview  = args.preview,
            history  = run
        )

        if not args.preview:
            Logger.info('Saving history')
            save_pkl(history, hfile)
            Logger.info('Done')
