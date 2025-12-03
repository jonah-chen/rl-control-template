#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --mem-per-cpu=6000M
#SBATCH --gpus-per-node=1
#SBATCH --time=11:59:59
#SBATCH --ntasks=22

module load python/3.12 swig rust clang cuda
module load scipy-stack
SLURM_TMPDIR="."
cp config.json $SLURM_TMPDIR/config.json
cp venv.tar.gz $SLURM_TMPDIR/venv.tar.gz
cd $SLURM_TMPDIR
tar -xzf ./venv.tar.gz .venv
source ./.venv/bin/activate
uv pip install "jax[cuda13]"
cd -
$SLURM_TMPDIR/.venv/bin/python /home/rougegod/projects/aip-amw8/rougegod/regit/rl-control-template/src/main.py -m
