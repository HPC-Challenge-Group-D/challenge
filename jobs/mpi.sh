#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=128

#SBATCH --time=01:30:00
#SBATCH --account=p200117
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=40
#SBATCH --cpus-per-task=1

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#load modules and navigate into directory
module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
cd ~/challenge

# compile
cd mpi
module load OpenMPI/4.1.4-NVHPC-22.7
mpicc -o jacobiSolver main.c solver.c -DNX=10000 -DNY=10000 -DMEASURETIME
srun ./jacobiSolver
