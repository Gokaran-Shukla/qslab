#!/bin/bash
#SBATCH --job-name=qslab-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8         # MPI ranks per node
#SBATCH --cpus-per-task=16          # threads per rank
#SBATCH --time=08:00:00
# Several nodes: qslab.pmap spreads tasks over MPI ranks; VMC/NQS/Majorana-MC chains
# are averaged with allreduce. Requires: pip install mpi4py
module load python openmpi  # adapt (Shaheen/Ibex: cray-mpich / openmpi modules)
export QSLAB_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores python mpi_vmc_and_ftlm.py
