#!/bin/bash

#SBATCH --job-name=myvenv
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=m23ss
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load anaconda/python3

CONDA_ROOT=/work/m23ss/m23ss/liyiyan/condaenvs2
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate myvenv

srun python <script>.py