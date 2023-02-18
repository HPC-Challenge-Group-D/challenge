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
TODO: This implementation should give a rough baseline on the implementaiton in CUDA
Final implementation should feature:
- Jacobi step as the only __global__ function
- Other functions as device funcitons called from the iteration
- Questionable stopping condition; Is the eps usefull to compute => Costly as to expensive and complicated CUDA reduction...
*/

__global__ void jacobiStepKernel(double *vp, double *v, double *f, int nx, int ny)
{
    
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    if( ixx < 1) 
        ixx+=sx;

    if(iyy < 1)
        iyy+=sy;

    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        for(int ix = ixx; ix < (nx-1); ix+=sx )
        {

            vp[iy*nx+ix] = -0.25 * (f[iy*nx+ix] -
                (v[nx*iy     + ix+1] + v[nx*iy     + ix-1] +
                    v[nx*(iy+1) + ix  ] + v[nx*(iy-1) + ix  ]));

        }
    }
}

__global__ void copyPointsKernel(double *src, double *dest, int nx, int ny)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    if( ixx < 1) 
        ixx+=sx;

    if(iyy < 1)
        iyy+=sy;
    
    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        for (int ix = ixx; ix < (nx-1); ix+=sx)
        {
            dest[nx*iy+ix] = src[nx*iy+ix];
        }
    }
}

__global__ void updateBoundaryKernel(double *vp, double *v, int nx, int ny)
{
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    int iyy = threadIdx.y + blockIdx.y * blockDim.y;
    int sy = blockDim.y * gridDim.y;

    if( ixx < 1) 
        ixx+=sx;

    if(iyy < 1)
        iyy+=sy;

    for (int ix = ixx; ix < (nx-1); ix+=sx)
    {
        v[nx*0      + ix] = v[nx*(ny-2) + ix];
        v[nx*(ny-1) + ix] = v[nx*1      + ix];
    }

    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        v[nx*iy + 0]      = v[nx*iy + (nx-2)];
        v[nx*iy + (nx-1)] = v[nx*iy + 1     ];
    }
}

int solver(double *v, double *f, int nx, int ny, double eps, int nmax)
{
    int n = 0;
    double e = 2. * eps;
    double *vp;

    dim3 threadsPerBlock;
    dim3 numberOfBlocks;

    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    threadsPerBlock = dim3(16,16);
    numberOfBlocks = dim3(4, 4);
    printf("%d, %d\n", props.warpSize, props.multiProcessorCount);

    cudaError_t errSync;
    cudaError_t asyncErr;
    
    //cudaMemPrefetchAsync(v, nx*ny*sizeof(double), deviceId);
    //cudaMemPrefetchAsync(vp, nx*ny*sizeof(double), deviceId);
    //cudaMemPrefetchAsync(f, nx*ny*sizeof(double), deviceId);

    //vp = (double *) malloc(nx * ny * sizeof(double));
    cudaMallocManaged(&vp, nx * ny * sizeof(double));

    //e > eps has been removed in order to speed up computation
    while ((n < nmax))
    {

        jacobiStepKernel<<<threadsPerBlock, numberOfBlocks>>>(vp, v, f, nx, ny);
        copyPointsKernel<<<threadsPerBlock, numberOfBlocks>>>(vp,v,nx,ny);
        updateBoundaryKernel<<<threadsPerBlock, numberOfBlocks>>>(vp,v,nx,ny);

        errSync = cudaGetLastError();
        if (errSync != cudaSuccess) { printf("Sync error: %s,\t%d\n", cudaGetErrorString(errSync), errSync); }
        asyncErr = cudaDeviceSynchronize();
        if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
        
        n++;
    }

    //Perform final iteration and finish by computing the error...

    jacobiStepKernel<<<threadsPerBlock, numberOfBlocks>>>(vp, v, f, nx, ny);
    errSync = cudaGetLastError();
    if (errSync != cudaSuccess) { printf("Sync error: %s,\t%d\n", cudaGetErrorString(errSync), errSync); }
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    e = 0.0;
    double w = 0.0;
    for (int ix = 1; ix < (nx-1); ix++)
    {
        for (int iy = 1; iy < (ny-1); iy++)
        {
            double d;
            d = fabs(vp[nx*iy+ix] - v[nx*iy+ix]);
            e = (d > e) ? d : e;
            w += fabs(v[nx*iy+ix]);
            v[nx*iy+ix] = vp[nx*iy+ix];
        }
    }

    for (int ix = 1; ix < (nx-1); ix++)
    {
        v[nx*0      + ix] = v[nx*(ny-2) + ix];
        v[nx*(ny-1) + ix] = v[nx*1      + ix];
        w += fabs(v[nx*0+ix]) + fabs(v[nx*(ny-1)+ix]);
    }

    for (int iy = 1; iy < (ny-1); iy++)
    {      
        v[nx*iy + 0]      = v[nx*iy + (nx-2)];
        v[nx*iy + (nx-1)] = v[nx*iy + 1     ];
        w += fabs(v[nx*iy+0]) + fabs(v[nx*iy+(nx-1)]);
    }

    w /= (nx * ny);
    e /= w;

    cudaFree(vp);

    if (e < eps)
        printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, nx, ny, e);
    else
        printf("ERROR: Failed to converge\n");

    return (e < eps ? 0 : 1);
}


