#!/bin/bash

#SBATCH --time=00:55:00
echo "script 1 started"
mkdir -p $SLURM_TMPDIR/$SLURM_JOB_ID

module --force purge
module load StdEnv/2023

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1

module load python/3.12 rust swig clang
uv venv $SLURM_TMPDIR/$SLURM_JOB_ID/.venv --cache-dir $SLURM_TMPDIR/$SLURM_JOB_ID/.cache --python 3.12
source $SLURM_TMPDIR/$SLURM_JOB_ID/.venv/bin/activate

uv pip install $path --cache-dir $SLURM_TMPDIR/$SLURM_JOB_ID/.cache

cd $SLURM_TMPDIR/$SLURM_JOB_ID
tar -cavf venv.tar.xz .venv
cp venv.tar.xz $path/
