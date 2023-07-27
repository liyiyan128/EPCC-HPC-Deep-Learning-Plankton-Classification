#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=m23ss

module load pytorch/1.13.1-gpu

CONDA_ROOT=/work/m23ss/m23ss/liyiyan/condaenvs2
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate myvenv

srun python <myscirpt>.py