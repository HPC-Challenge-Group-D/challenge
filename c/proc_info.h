/*
 * Header proc_info.h
 * Defines a struct for storing process information
 * Also the system parameters for the execution are set here
 */
#pragma once
#include <mpi.h>

/*
 * Define the directions for the MPI Cartesian communication
 */
#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3

/*
 * Define execution parameters
 */
#ifndef NX
#define NX 1024
#endif
#ifndef NY
#define NY 16
#endif
#define NMAX 200000
#define EPS 1e-5

#define INTERVAL_ERROR_CHECK 100

/*
 * Process info struct storing all relevant information for the current process.
 */
struct proc_info {
    int rank;   /*Rank of the current process*/
    int size;   /*Number of processes in total*/
    int coords[2];  /*Coordinates of the current process in the cartesian grid*/
    int neighbors[4];   /*Neighbors of the process in the cartesian grid*/
    int dims[2];    /*Dimension of the cartesian grid*/
    MPI_Comm cartcomm; /*Cartesian communicator*/
    MPI_Datatype row, column; /*MPI Types for communicating a row or a column*/
};
