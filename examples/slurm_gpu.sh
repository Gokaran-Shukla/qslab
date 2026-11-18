#!/bin/bash
#SBATCH --job-name=qslab-gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# Optional GPU backend. Requires: pip install cupy-cuda12x (match your CUDA)
module load cuda python
export QSLAB_BACKEND=gpu
python gpu_ed_dmrg.py
