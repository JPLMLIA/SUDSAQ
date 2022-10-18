#!/bin/bash

export JOBLIB_TEMP_FOLDER=/scratch/jamesmo/tmp
export HDF5_USE_FILE_LOCKING=FALSE

source /usr/local/anaconda3/bin/activate
conda activate sudsaq

cd ~/suds-air-quality/sudsaq/


configs=(
  "configs/dev/ml/monthly/bias/tz/8hr.median.yml"
  "configs/dev/ml/monthly/bias/utc/8hr.median.yml"
  "configs/dev/ml/monthly/bias/utc/8hr.mean.yml"
  "configs/dev/ml/monthly/bias/tz/8hr.mean.yml"
  "configs/dev/ml/monthly/bias/utc/24hr.mean.yml"
)
sections=(
  "jan"
  "jul"
  "feb"
  "mar"
  "apr"
  "may"
  "jun"
  "aug"
  "sep"
  "oct"
  "nov"
  "dec"
)

for config in ${configs[@]}; do
  echo "Running config: $config"
  for section in ${sections[@]}; do
    echo "Running section: $section"
    nice -n 19 python ml/create.py -c $config -s $section
  done
done

echo "Done"
