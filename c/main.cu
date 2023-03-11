//
// Created by Patrik Rac on 10.03.23.
//

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

/*
 * MPI-CUDA version of the Jacobi solver
 * Code runs on multiple GPUs and is used to analyze the behavior and of
 * the program in this setting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "proc_info.h"

/*External solver function*/
int solver(double *, double *, int, int, double, int, struct proc_info*);

/*Internal helper functions*/
void initDevice();
static int computeOptimalPartitioning(int nx, int ny, int size);
void initHost(double *, double *, int, int, int, int);

int main()
{


    /*
     * Setup Phase
     */
    /*Set the appropriate device before the call to MPI_init()*/
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
        fflush(stdout);
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

    /*Allocate grid and data on the Device*/
    double *v;
    double *f;

    // Allocate memory
    cudaMalloc(&v, local_nx * local_ny * sizeof(double));
    cudaMalloc(&f, local_nx * local_ny * sizeof(double));

    // Initialise data
    initHost(v,f, local_nx, local_ny, x_offset, y_offset);

    cudaDeviceSynchronize();

    /*Start timer*/
    struct timespec ts;
    double start, end;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;

    // Call solver
    solver(v, f, local_nx, local_ny, EPS, NMAX, &proc);

    clock_gettime(CLOCK_MONOTONIC, &ts);
    end = (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
    printf("Execution time: %f s\n", end-start);

    //for (int iy = 0; iy < NY; iy++)
    //    for (int ix = 0; ix < NX; ix++)
    //        printf("%d,%d,%e\n", ix, iy, v[iy*NX+ix]);

    // Clean-up
    cudaFree(v);
    cudaFree(f);

    return 0;
}

/*
 * Kernel funcition to initialize the
 */
__global__
void initKernel(double *v, double *f, int nx, int ny, int offset_x, int offset_y)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    // Initialise input
    for (int iy = iyy; iy < ny; iy+=sy)
        for (int ix = ixx; ix < nx; ix+=sx)
        {
            v[nx*iy+ix] = 0.0;

            const double x = 2.0 * (ix+offset_x) / (NX - 1.0) - 1.0;
            const double y = 2.0 * (iy+offset_y) / (NY - 1.0) - 1.0;
            f[nx*iy+ix] = sin(x + y);
        }
}


void initHost(double *v, double *f, int nx, int ny, int offset_x, int offset_y)
{
    dim3 threadsPerBlock;
    dim3 numberOfBlocks;

    threadsPerBlock = dim3(16, 16);
    numberOfBlocks = dim3(8,8);

    initKernel<<<numberOfBlocks, threadsPerBlock>>>(v, f, nx, ny, offset_x, offset_y);
}

void initDevice()
{
    char * localRankStr = NULL;
    int local_rank = 0;

    // We extract the local rank initialization using an environment variable
    if ((localRankStr = getenv("SLURM_LOCALID")) != NULL)
    {
        local_rank = atoi(localRankStr);
    }
    else
    {
        printf("Could not determine the appropriate local rank!\n");
    }

    printf("Initializing with device %d\n", local_rank);
    fflush(stdout);
    cudaSetDevice(local_rank);
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

