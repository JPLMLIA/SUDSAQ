#!/bin/bash

echo "Installing SUDSAQ requirements"

mamba env create -f environment.yml
mamba activate sudsaq

echo "Installing SUDSAQ"
pip install -e .

echo "Done"
