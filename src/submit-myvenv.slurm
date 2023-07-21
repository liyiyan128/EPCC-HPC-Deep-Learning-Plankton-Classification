#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=m23ss

module load anaconda/python3
module load nvidia/nvhpc
module load nvidia/cudnn

CONDA_ROOT=/work/m23ss/m23ss/liyiyan/condaenvs2
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate myvenv

srun python <myscript>.py