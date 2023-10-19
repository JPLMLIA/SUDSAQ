FROM mambaorg/micromamba:1.5.1

USER root
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      g++

USER $MAMBA_USER
WORKDIR /home/$MAMBA_USER

# Copy and install the project
COPY --chown=$MAMBA_USER:$MAMBA_USER . suds-air-quality

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -n base -c conda-forge jupyterlab nb_conda_kernels &&\
    micromamba install -y -n base -f suds-air-quality/envs/environment.yml &&\
    pip install suds-air-quality/envs/mlky-2023.10.0-py3-none-any.whl &&\
    pip install -e suds-air-quality

# Explicitly set the shell to bash so the Jupyter server defaults to it
ENV SHELL=/bin/bash

# Start the Jupyterlab server
EXPOSE 8888
CMD jupyter-lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''


TOAR:
- reprocess v2 data:
  - mean, median, stddev
  need gridded data
pipeline:
- do extended 10 year runs
