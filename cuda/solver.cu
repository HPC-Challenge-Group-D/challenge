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

/*
TODO: This implementation should give a rough baseline on the implementation in CUDA
 * Non optimal CUDA implementation
 * The main bottleneck of this implementation is the constant transfer of the weights and errors
 * They are transferred to the CPU for assembly and evaluation each iteration.
 * This can possibly be solved using a well implemented GPU reduction (HOWEVER tricky).
 * Furthermore, more care can be placed on the boundary value placement and synchronisation.
 */


/*
 * Kernel to perform the Jacobi Step
 * Results of the steps are then rewritten into the original array
 * The Kernel calculates its own error and weight value at position tid
 */
__global__ void jacobiStepKernel(double *vp, double *v, double *f, int nx, int ny, double *e, double *w)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
    const unsigned int tid = blockNumInGrid * threadsPerBlock + threadNumInBlock;

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
            v[nx*iy+ix] = vp[nx*iy+ix];
            w[tid] += fabs(v[nx*iy+ix]);
        }
    }
}


/*
 * Kernel to update the boundary points.
 * More care should here be placed on their placement...
 */
__global__ void updateBoundaryKernel(double *v, int nx, int ny, double *w)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;


    /*
     * Update the boundary points
     */
    for (int ix = idx+1; ix < (nx-1); ix+=stride)
    {
        v[nx*0      + ix] = v[nx*(ny-2) + ix];
        v[nx*(ny-1) + ix] = v[nx*1      + ix];
        w[idx] += fabs(v[nx*0+ix]) + fabs(v[nx*(ny-1)+ix]);
    }

    for (int iy = idx+1; iy < (ny-1); iy+=stride)
    {
        v[nx*iy + 0]      = v[nx*iy + (nx-2)];
        v[nx*iy + (nx-1)] = v[nx*iy + 1     ];
        w[idx] += fabs(v[nx*iy+0]) + fabs(v[nx*iy+(nx-1)]);
    }
}

/*
 * Host solver methods, which handles synchronisation and other important tasks.
 * Furthermore, it is responsible for assembling the error and weight necessary for the stopping condition...
 */
int solver(double *v, double *f, int nx, int ny, double eps, int nmax)
{
    int n = 0;
    double e = 2. * eps;
    double *vp;

    double *w_device, *e_device;

    dim3 threadsPerBlock;
    dim3 numberOfBlocks;

    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    threadsPerBlock = dim3(4,4);
    numberOfBlocks = dim3(2, 2);

    printf("The program is running on a GPU with %d warps, %d MPs\n", props.warpSize, props.multiProcessorCount);

    cudaMallocManaged(&vp, nx * ny * sizeof(double));

    const unsigned int num_gpu_threads = numberOfBlocks.x * numberOfBlocks.y * threadsPerBlock.x * threadsPerBlock.y;

    cudaMallocManaged(&w_device, num_gpu_threads*sizeof(double));
    cudaMallocManaged(&e_device, num_gpu_threads*sizeof(double));

    cudaError_t errSync;
    cudaError_t asyncErr;
    
    cudaMemPrefetchAsync(v, nx*ny*sizeof(double), deviceId);
    cudaMemPrefetchAsync(vp, nx*ny*sizeof(double), deviceId);
    cudaMemPrefetchAsync(f, nx*ny*sizeof(double), deviceId);

    //e > eps has been removed in order to speed up computation
    while ((e > eps) && (n < nmax))
    {
        cudaMemPrefetchAsync(w_device, num_gpu_threads*sizeof(double), deviceId);
        cudaMemPrefetchAsync(e_device, num_gpu_threads*sizeof(double), deviceId);

        jacobiStepKernel<<<threadsPerBlock, numberOfBlocks>>>(vp, v, f, nx, ny, e_device, w_device);
        cudaDeviceSynchronize();
        updateBoundaryKernel<<<threadsPerBlock.x*threadsPerBlock.y, numberOfBlocks.x*numberOfBlocks.y>>>(v, nx, ny, w_device);

        errSync = cudaGetLastError();
        if (errSync != cudaSuccess) { printf("Sync error: %s,\t%d\n", cudaGetErrorString(errSync), errSync); }
        asyncErr = cudaDeviceSynchronize();
        if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

        cudaMemPrefetchAsync(w_device, num_gpu_threads*sizeof(double), cudaCpuDeviceId);
        cudaMemPrefetchAsync(e_device, num_gpu_threads*sizeof(double), cudaCpuDeviceId);

        e = 0.0;
        double w = 0.0;

        for (int i = 0; i < num_gpu_threads; i++)
        {
            w += w_device[i];
            e = (e_device[i] > e) ? e_device[i] : e;
        }

        w /= (nx * ny);
        e /= w;

        //if ((n % 10) == 0)
            //printf("%5d, %0.4e\n", n, e, w);
        
        n++;
    }

    cudaFree(vp);
    cudaFree(w_device);
    cudaFree(e_device);

    if (e < eps)
        printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e);
    else
        printf("ERROR: Failed to converge\n");

    return (e < eps ? 0 : 1);
}


