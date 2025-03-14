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
 * Pure CUDA version of the Jacobi solver
 * Code runs on single GPU and is used to analyze the behavior and of
 * the program in this setting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef NX
#define NX 1024
#endif
#ifndef NY
#define NY 16
#endif
#define NMAX 200000
#define EPS 1e-5

int solver(double *, double *, int, int, double, int);

__host__
void initHost(double *, double *);

int main()
{
    double *v;
    double *f;

    // Allocate memory
    cudaMalloc(&v, NX * NY * sizeof(double));
    cudaMalloc(&f, NX * NY * sizeof(double));

    // Initialise input
    initHost(v,f);

    /*Start timer*/
    struct timespec ts;
    double start, end;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;

    // Call solver
    solver(v, f, NX, NY, EPS, NMAX);

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

__global__
void initKernel(double *v, double *f)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    // Initialise input
    for (int iy = iyy; iy < NY; iy+=sy)
        for (int ix = ixx; ix < NX; ix+=sx)
        {
            v[NX*iy+ix] = 0.0;

            const double x = 2.0 * ix / (NX - 1.0) - 1.0;
            const double y = 2.0 * iy / (NY - 1.0) - 1.0;
            f[NX*iy+ix] = sin(x + y);
        }
}


__host__
void initHost(double *v, double *f)
{
    dim3 threadsPerBlock;
    dim3 numberOfBlocks;

    threadsPerBlock = dim3(16, 16);
    numberOfBlocks = dim3(8,8);

    initKernel<<<numberOfBlocks, threadsPerBlock>>>(v, f);
}

