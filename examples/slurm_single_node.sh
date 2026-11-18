#!/bin/bash
#SBATCH --job-name=qslab-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # all cores of the node
#SBATCH --time=04:00:00
#SBATCH --mem=0
# One node, shared memory: Numba kernels + BLAS threads + qslab.pmap process pool.
module load python  # adapt to your cluster
export QSLAB_NUM_THREADS=$SLURM_CPUS_PER_TASK
export QSLAB_NUM_JOBS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python ed_large_kagome.py
