#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --account=p200117
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#load modules and navigate into directory
module load foss/2021a
cd ~/challenge

# compile
mpicc -o program main.c solver.c
mpirun -np 4 program