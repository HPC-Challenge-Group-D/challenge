# MPI_OpenMP version

## Compile

Defaults are NX=1024, NY=1024
Set MEASURETIME to record walltime.

```bash
module load OpenMPI / OpenMPI/4.1.4-NVHPC-22.7
mpicc -o mpi_omp_program main.c solver.c [opt.: -DNX=1024 -DNY=32 -DMEASURETIME ]
```

## Run

```bash
srun ./mpi_omp_program              # runs with all available processes
```