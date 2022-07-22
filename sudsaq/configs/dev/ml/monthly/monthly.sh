#!/bin/bash

source /usr/local/anaconda3/bin/activate
conda activate sudsaq

cd ~/suds-air-quality/sudsaq/

config="configs/dev/ml/monthly/bias.11-14.yml"
sections=(
  "jan"
  "feb"
  "mar"
  "apr"
  "may"
  "jun"
  "jul"
  "aug"
  "sep"
  "oct"
  "nov"
  "dec"
)

for section in ${sections[@]}; do
  echo "Running $section"
  python ml/create.py -c $config -s $section
done

echo "Done"
