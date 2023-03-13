#pragma once 
#include <mpi.h>

#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3

#ifndef NX
#define NX 256
#endif
#ifndef NY
#define NY 256
#endif
#define NMAX 200000
#define EPS 1e-5

struct proc_info {
    int rank;   /*Rank of the current process*/
    int size;   /*Number of processes in total*/
    int coords[2];  /*Coordinates of the current process in the cartesian grid*/
    int neighbors[4];   /*Neighbors of the process in the cartesian grid*/
    int dims[2];    /*Dimension of the cartesian grid*/
    MPI_Comm cartcomm; /*Cartesian communicator*/
    MPI_Datatype row, column; /*MPI Types for communicating a row or a column*/
};