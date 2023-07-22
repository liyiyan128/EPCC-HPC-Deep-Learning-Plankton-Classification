#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --account=m23ss

module load python/3.10.8-gpu
# module load anaconda/python3
# module load nvidia/nvhpc/22.11
# module load nvidia/cudnn/8.6.0-cuda-11.6

CONDA_ROOT=/work/m23ss/m23ss/liyiyan/condaenvs2
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate myvenv

srun python <myscript>.py