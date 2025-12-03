#!/bin/bash

module load python/3.12

# make sure home folder has a venv
if [ ! -d ~/.venv ]; then
  echo "making a new virtual env in ~/.venv"
  cd ~
  uv venv --python 3.12
fi

source ~/.venv/bin/activate
echo "installing PyExpUtils"
uv pip install PyExpUtils-andnp

path="."
cd $path

echo "scheduling a job to install project dependencies"
sbatch --ntasks=1 --mem-per-cpu="12G" --export=path="$(pwd)" ./scripts/local_node_venv.sh
#sbatch --ntasks=1 --mem-per-cpu="12G" --export=path="$(pwd)" ./scripts/local_node_venv_gpu.sh
