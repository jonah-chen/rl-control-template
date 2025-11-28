#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --mem-per-cpu=16000M
#SBATCH --time=7:59:59
#SBATCH --ntasks=12

module load python/3.12 swig rust clang
module load scipy-stack
cp config.json $SLURM_TMPDIR/config.json
cp venv.tar.gz $SLURM_TMPDIR/venv.tar.gz
cd $SLURM_TMPDIR
tar -xzf ./venv.tar.gz .venv
source ./.venv/bin/activate
cd -
$SLURM_TMPDIR/.venv/bin/python /home/rougegod/projects/aip-amw8/rougegod/regit/rl-control-template/src/main.py -m
