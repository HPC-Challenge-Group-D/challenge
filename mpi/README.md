# MPI-only version

## Compile

Defaults are NX=1024, NY=1024
Set MEASURETIME to record walltime.

```bash
module load OpenMPI / OpenMPI/4.1.4-NVHPC-22.7
mpicc -o mpi_program main.c solver.c [opt.: -DNX=1024 -DNY=32 -DMEASURETIME ]
```

## Run

```bash
srun ./mpi_program              # runs with all available processes
mpirun -np <n> ./mpi_program    # runs with specified n processes
```