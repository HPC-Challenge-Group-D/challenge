/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * This source code is in parts based on code from Jiri Kraus (NVIDIA) and
 * Andreas Herten (Forschungszentrum Juelich)
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND Any
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR Any DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON Any THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN Any WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cub/device/device_reduce.cuh>

#ifndef INTERVAL_ERROR_CHECK
#define INTERVAL_ERROR_CHECK 100
#endif

/*Get headers of the device reduction functions*/
void deviceReduce(double *in, double* out, int N);
void deviceReduceMax(double *in, double* out, int N);

/*
 * Kernel to perform the Jacobi Step
 * Results of the steps are then rewritten into the original array
 * The Device calculates its own error and weight value at position tid
 */
__global__
void jacobiStepKernel(double *vp, double *v, double *f, int nx, int ny, double *e, double *w)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    /*Thread id for computation of the local weights and errors*/
    const unsigned int tid = ixx*sy + iyy;

    e[tid] = 0.0;
    w[tid] = 0.0;


    if(ixx < 1)
        ixx+=sx;

    if(iyy < 1)
        iyy+=sy;

    /*
     * Perform actual Jacobi Step
     */
    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        for(int ix = ixx; ix < (nx-1); ix+=sx)
        {
            double d;

            vp[iy*nx+ix] = -0.25 * (f[iy*nx+ix] -
                (v[nx*iy     + ix+1] + v[nx*iy     + ix-1] +
                    v[nx*(iy+1) + ix  ] + v[nx*(iy-1) + ix  ]));

            d = fabs(vp[nx*iy+ix] - v[nx*iy+ix]);
            e[tid] = (d > e[tid]) ? d : e[tid];
        }
    }
    /*
     * Copy the points on to the original array
     */
    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        for (int ix = ixx; ix < (nx-1); ix+=sx)
        {
            //v[nx*iy+ix] = vp[nx*iy+ix];
            w[tid] += fabs(vp[nx*iy+ix]);
        }
    }
}


/*
 * Kernel to update the boundary points.
 * More care should here be placed on their placement...
 */
__global__
void updateBoundaryKernel(double *v, int nx, int ny, double *w)
{
    /*
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;*/

    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    /*Thread id for computation of the local weights and errors*/
    const unsigned int tid = ixx*sy + iyy;

    if(ixx < 1)
        ixx+=sx;

    if(iyy < 1)
        iyy+=sy;

    /*
     * Update the boundary points
     */
    if(iyy == sx)
        for (int ix = ixx; ix < (nx-1); ix+=sx)
        {
            v[nx*0      + ix] = v[nx*(ny-2) + ix];
            v[nx*(ny-1) + ix] = v[nx*1      + ix];
            w[tid] += fabs(v[nx*0+ix]) + fabs(v[nx*(ny-1)+ix]);
        }

    if(ixx == sy)
        for (int iy = iyy; iy < (ny-1); iy+=sy)
        {
            v[nx*iy + 0]      = v[nx*iy + (nx-2)];
            v[nx*iy + (nx-1)] = v[nx*iy + 1     ];
            w[tid] += fabs(v[nx*iy+0]) + fabs(v[nx*iy+(nx-1)]);
        }
}

/*
 * Host solver methods, which handles synchronisation and other important tasks.
 */
__host__
int solver(double *v, double *f, int nx, int ny, double eps, int nmax)
{
    int n = 0;

    /*Allocate memory for the secondary array vp*/
    double *vp;
    cudaMalloc(&vp, nx * ny * sizeof(double));

    /*Set the number of blocks and number of Threads for the kernel launches*/
    dim3 threadsPerBlock;
    dim3 numberOfBlocks;
    threadsPerBlock = dim3(16, 16);
    numberOfBlocks = dim3(16,16);

    /*Calculate theoretical value of total number of gpu threads*/
    const unsigned int num_gpu_threads = (numberOfBlocks.x * threadsPerBlock.x) * (numberOfBlocks.y * threadsPerBlock.y);

    /*Allocate local array for the errors and weights on device*/
    double *w_device, *e_device;
    cudaMalloc(&w_device, num_gpu_threads*sizeof(double));
    cudaMalloc(&e_device, num_gpu_threads*sizeof(double));

    /*Allocate memory for the resulting reduced weight and error*/
    double *e, *w;
    cudaMallocManaged(&w, sizeof(double));
    cudaMallocManaged(&e, sizeof(double));
    *e = 2. * eps;

    void *sum_temp_storage=NULL;
    size_t sum_temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(sum_temp_storage, sum_temp_storage_bytes, w_device, w, num_gpu_threads);
    cudaMalloc(&sum_temp_storage,sum_temp_storage_bytes);

    void *max_temp_storage = NULL;
    size_t max_temp_storage_bytes = 0;
    cub::DeviceReduce::Max(max_temp_storage, max_temp_storage_bytes, e_device, e, num_gpu_threads);
    cudaMalloc(&max_temp_storage, max_temp_storage_bytes);

    cudaDeviceSynchronize();

    while ((e[0] > eps) && (n < nmax))
    {
        //cudaMemPrefetchAsync(v, nx*ny*sizeof(double), deviceId);

        jacobiStepKernel<<<numberOfBlocks, threadsPerBlock>>>(vp, v, f, nx, ny, e_device, w_device);

        /*Swap the pointer for the primary and secondary array before updating the boundary*/
        double *tmp = v;
        v = vp;
        vp = tmp;

        updateBoundaryKernel<<<numberOfBlocks, threadsPerBlock>>>(v, nx, ny, w_device);

        if((n+1)%INTERVAL_ERROR_CHECK == 0)
        {
            *w = 0;
            *e = 0;
            //deviceReduce(w_device, w, num_gpu_threads);
            cub::DeviceReduce::Sum(sum_temp_storage, sum_temp_storage_bytes, w_device, w, num_gpu_threads);
            //deviceReduceMax(e_device, e, num_gpu_threads);
            cub::DeviceReduce::Max(max_temp_storage, max_temp_storage_bytes, e_device, e, num_gpu_threads);

            cudaDeviceSynchronize();
            w[0] /= (nx * ny);
            e[0] /= w[0];
        }


        //cudaMemPrefetchAsync(v, nx*ny*sizeof(double), cudaCpuDeviceId);

        //if ((n % 10) == 0)
            //printf("%5d, %0.4e,  %0.4e\n", n, e[0], w[0]);

        n++;
    }

    /*Last "safeguard synchronisation before end of the program"*/
    cudaDeviceSynchronize();

    /*Clean-up*/
    cudaFree(vp);
    cudaFree(w_device);
    cudaFree(e_device);

    cudaFree(sum_temp_storage);
    cudaFree(max_temp_storage);

    double e_local = e[0];
    cudaFree(e);
    cudaFree(w);

    if (e_local < eps)
        printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e_local);
    else
        printf("ERROR: Failed to converge\n");

    return (e_local < eps ? 0 : 1);
}


