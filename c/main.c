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
#include <time.h>
#include <math.h>

#include <mpi.h>

#include "proc_info.h"

#include "jacobi.h"
#include "init.h"
#include "reduction.h"

/*
 * Parallel solver
 */
int solver(double *, double *, double *, int, int, double, int, struct proc_info *);

/*
 * Helper function that calculates the optimal partitioning of processes for the current domain.
 * Takes into account the xy-ration of the domain of the problem to place processes.
 */
static int computeOptimalPartitioning(int nx, int ny, int size);


int main()
{
    /*
     * Setup Phase
     */
    initDevice();

    /*Initialize MPI and the process info struct*/
    MPI_Init(NULL, NULL);
    struct proc_info proc;

    MPI_Comm_size(MPI_COMM_WORLD, &proc.size);

    /*Set-up proper distribution of the grid among a 2D process arrangement*/
    proc.dims[1] = computeOptimalPartitioning(NX,NY, proc.size);
    proc.dims[0] = proc.size/proc.dims[1];

    /*Set up the cartesian communicator using wraparound boundaries*/
    int periods[2] = {1,1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc.dims, periods, 1, &proc.cartcomm);

    MPI_Comm_rank(proc.cartcomm, &proc.rank);

    if(proc.rank == 0)
    {
        printf("Running MPI-CUDA with %d processes\n", proc.size);
    }

    /*Get the ranks of the neighboring processes in all directions*/
    MPI_Cart_shift(proc.cartcomm, 0, 1, &proc.neighbors[SOUTH], &proc.neighbors[NORTH]);
    MPI_Cart_shift(proc.cartcomm, 1, 1, &proc.neighbors[WEST], &proc.neighbors[EAST]);

    /*Get own cartesian coordinates*/
    MPI_Cart_coords(proc.cartcomm, proc.rank, 2, proc.coords);

    /*Local size of the grid; Additional rows and columns for boundary points*/
    size_t local_nx = ((NX-2)/proc.dims[1]) + 2;
    size_t local_ny = ((NY-2)/proc.dims[0]) + 2;

    int y_offset = (local_ny - 2) * proc.coords[0];
    int x_offset = (local_nx - 2) * proc.coords[1];

    /*
     * If the grid points are not divisible my the processor dimension.
     * The last processor on the axis will receive the remaining points.
     */
    if (proc.coords[0] == proc.dims[0] - 1)
    {
        local_ny += ((NY - 2) % proc.dims[0]);
    }

    if (proc.coords[1] == proc.dims[1] - 1)
    {
        local_nx += ((NX - 2) % proc.dims[1]);
    }

    /*MPI Datatype for the communication of the boundary points between processes*/
    MPI_Type_vector(local_nx-2, 1, 1, MPI_DOUBLE, &proc.row);
    MPI_Type_commit(&proc.row);

    MPI_Type_vector(local_ny-2, 1, local_nx, MPI_DOUBLE, &proc.column);
    MPI_Type_commit(&proc.column);

    /*
     * End of setup phase
     */

    double *v;
    double *f;
    double *vp;

    // Allocate memory
    //v = (double *) malloc(local_nx * local_ny * sizeof(double));
    //f = (double *) malloc(local_nx * local_ny * sizeof(double));

    //Allocate memory on the device
    prepareDataMemory(&v, &vp, &f, local_nx, local_ny, x_offset, y_offset);

    /*Start timer*/
    struct timespec ts;
    double start, end;
    if(proc.rank == 0)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start = (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
    }


    // Call solver
    solver(v, vp, f, local_nx, local_ny, EPS, NMAX, &proc);


    /*End timer*/
    if(proc.rank == 0)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        end = (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
        printf("Execution time: %f s\n", end-start);
    }

    //for (int iy = 0; iy < NY; iy++)
    //    for (int ix = 0; ix < NX; ix++)
    //        printf("%d,%d,%e\n", ix, iy, v[iy*NX+ix]);

    // Clean-up
    freeDataMemory(&v, &vp, &f);

    MPI_Type_free(&proc.row);
    MPI_Type_free(&proc.column);
    MPI_Finalize();

    return 0;
}


static int computeOptimalPartitioning(int nx, int ny, int size)
{
    const double xy_ratio = ((double) nx)/ny; 
    double best_ratio_distance = INFINITY;
    int best_procX = 0;
    for (int d = 1; d <= size; d++)
    {
        if(size%d == 0)
        {
            const int div2 = size/d;
            const double ratio = ((double )d) / div2;
            if (fabs(ratio - xy_ratio) < best_ratio_distance)
            {
                best_procX = d;
                best_ratio_distance = fabs(ratio - xy_ratio);
            }
        }
    }
    return best_procX;
}


int solver(double *v, double *vp, double *f, int nx, int ny, double eps, int nmax, struct proc_info *proc)
{
    int n = 0;
    //double e = 2. * eps;

    double *w_device, *e_device;
    double *e, *w;

    unsigned int num_gpu_threads = prepareMiscMemory(&w, &e, &w_device, &e_device);

    *e = 2. * eps;

    while (((*e) > eps) && (n < nmax))
    {

        /*Start of computation phase*/
        jacobiStep(vp, v, f, nx, ny, e_device, w_device);
        sync();
        /*End of computation phase*/

        /*Communication Phase*/
        //TODO: Maybe optimize here...

        MPI_Sendrecv(&v[nx*(ny-2)+1], 1, proc->row, proc->neighbors[NORTH], 0,
                     &v[1], 1, proc->row, proc->neighbors[SOUTH], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx*1 + 1], 1, proc->row, proc->neighbors[SOUTH], 0,
                     &v[nx*(ny-1) + 1], 1, proc->row, proc->neighbors[NORTH], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx*2 - 2], 1, proc->column, proc->neighbors[EAST], 0,
                     &v[nx], 1, proc->column, proc->neighbors[WEST], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx+1], 1, proc->column, proc->neighbors[WEST], 0,
                     &v[nx*2 - 1], 1, proc->column, proc->neighbors[EAST], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        /*End of communication phase*/
        if(n % INTERVAL_ERROR_CHECK == 0)
        {
            *w = 0;
            *e = 0;
            /*Compute weight on the boundary*/
            if (proc->coords[0] == 0)
            {
                weightBoundary_x(v,nx,ny,w_device,0);
            }

            if(proc->coords[0] == proc->dims[0]-1)
            {
                weightBoundary_x(v,nx,ny,w_device,ny-1);
            }

            if(proc->coords[1] == 0)
            {
                weightBoundary_y(v,nx,ny,w_device,0);
            }

            if(proc->coords[1] == proc->dims[1]-1)
            {
                weightBoundary_y(v,nx,ny,w_device,nx-1);
            }

            deviceReduce(w_device, w, num_gpu_threads);
            deviceReduceMax(e_device, e, num_gpu_threads);
            sync();

            MPI_Allreduce(MPI_IN_PLACE, e, 1, MPI_DOUBLE, MPI_MAX, proc->cartcomm);
            MPI_Allreduce(MPI_IN_PLACE, w, 1, MPI_DOUBLE, MPI_SUM, proc->cartcomm);

            *w /= (NX * NY);
            *e /= *w;
        }
        /*
        if(proc->rank == 0)
        {
            if ((n % 10) == 0)
                printf("%5d, %0.4e\n", n, e);
        }*/
        //if ((n % 10) == 0)
        //    printf("%5d, %0.4e\n", n, e);

        n++;
    }

    double e_local = *e;

    freeMiscMemory(&w,&e,&w_device,&e_device);


    if(proc->rank == 0)
    {
        if (e_local < eps)
            printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, NX, NY, e_local);
        else
            printf("ERROR: Failed to converge\n");
    }

    return (e_local < eps ? 0 : 1);
}
