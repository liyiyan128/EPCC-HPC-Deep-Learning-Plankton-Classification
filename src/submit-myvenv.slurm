#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=m23ss

# module load python/3.10.8-gpu
module load anaconda/python3
module load nvidia/nvhpc/22.2
module load nvidia/cudnn/8.6.0-cuda-11.6

CONDA_ROOT=/work/m23ss/m23ss/liyiyan/condaenvs2
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate myvenv

srun python alexnet_transfer_learning.py