/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>
#include "proc_info.h"

int solver(double *, double *, int, int, double, int, struct proc_info *);

int main()
{
    /*Setup*/
    MPI_Init(NULL, NULL);
    struct proc_info proc;

    MPI_Comm_size(MPI_COMM_WORLD, &proc.size);

    //TODO: Find proper distribution considering the grid size ...
    /*WARNING: Current setup only works if size iis a square*/
    /*Set-up proper distribution of the grid among a 2D process arangement*/
    proc.dims[0] = (proc.size/sqrt(proc.size));
    proc.dims[1] = (proc.size/sqrt(proc.size));

    int periods[2] = {1,1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc.dims, periods, 1, &proc.cartcomm);

    MPI_Comm_rank(proc.cartcomm, &proc.rank);
    
    MPI_Cart_shift(proc.cartcomm, 0, 1, &proc.neighbors[SOUTH], &proc.neighbors[NORTH]);
    MPI_Cart_shift(proc.cartcomm, 1, 1, &proc.neighbors[WEST], &proc.neighbors[EAST]);

    MPI_Cart_coords(proc.cartcomm, proc.rank, 2, proc.coords);

    //TODO: Compute extra points for last processes if number is not divisable ... 
    /*Local size of the grid; Additional rows and columns for boundary points*/
    size_t local_nx = ((NX-2)/proc.dims[1]) + 2;
    size_t local_ny = ((NY-2)/proc.dims[0]) + 2;

    int y_offset = (local_ny - 2) * proc.coords[0];
    int x_offset = (local_nx - 2) * proc.coords[1];

    /*MPI Datatypes for the communication of the boundary points inbetween processes*/
    MPI_Type_vector(local_nx-2, 1, 1, MPI_DOUBLE, &proc.row);
    MPI_Type_commit(&proc.row);

    MPI_Type_vector(local_ny-2, 1, local_nx, MPI_DOUBLE, &proc.column);
    MPI_Type_commit(&proc.column);

    double *v;
    double *f;

    // Allocate memory
    v = (double *) malloc(local_nx * local_ny * sizeof(double));
    f = (double *) malloc(local_nx * local_ny * sizeof(double));

    // Initialise input
    for (int iy = 1; iy < local_ny-1; iy++)
        for (int ix = 1; ix < local_nx-1; ix++)
        {
            v[local_nx*iy+ix] = 0.0;

            const double x = 2.0 * (ix+x_offset) / (NX - 1.0) - 1.0;
            const double y = 2.0 * (iy+y_offset) / (NY - 1.0) - 1.0;
            f[local_nx*iy+ix] = sin(x + y);
        }

    // Call solver
    solver(v, f, local_nx, local_ny, EPS, NMAX, &proc);

    //for (int iy = 0; iy < NY; iy++)
    //    for (int ix = 0; ix < NX; ix++)
    //        printf("%d,%d,%e\n", ix, iy, v[iy*NX+ix]);

    // Clean-up
    free(v);
    free(f);

    MPI_Type_free(&proc.row);
    MPI_Type_free(&proc.column);
    MPI_Finalize();

    return 0;
}