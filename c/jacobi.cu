//
// Created by Patrik Rac on 08.03.23.
//
#include "jacobi.h"

/*
 * Kernel to perform the Jacobi Step
 * Results of the steps are then rewritten into the original array
 * The Kernel calculates its own error and weight value at position tid
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
            v[nx*iy+ix] = vp[nx*iy+ix];
            w[tid] += fabs(v[nx*iy+ix]);
        }
    }
}

__global__
void weightBoundaryKernel_x(double *v, int nx, int ny, double *w, int iy)
{
    /*
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;*/
    int ixx = threadIdx.x + blockIdx.x * blockDim.x;
    int sx = blockDim.x * gridDim.x;

    /*Thread id for computation of the local weights and errors*/
    const unsigned int tid = ixx;

    if(ixx < 1)
        ixx+=sx;

    /*
     * Update the boundary points
     */
    for (int ix = ixx; ix < (nx-1); ix+=sx)
    {
        //v[nx*1      + ix] = v[nx*0     + ix];
        w[tid] += fabs(v[nx*iy + ix]);
    }
}

__global__
void weightBoundaryKernel_y(double *v, int nx, int ny, double *w, int ix)
{
    /*
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;*/
    int iyy = threadIdx.x + blockIdx.x * blockDim.x;
    int sy = blockDim.x * gridDim.x;

    /*Thread id for computation of the local weights and errors*/
    const unsigned int tid = iyy;

    if(iyy < 1)
        iyy+=sy;

    /*
     * Update the boundary points
     */
    for (int iy = iyy; iy < (ny-1); iy+=sy)
    {
        //v[nx*iy + 1]      = v[nx*iy + 0];
        w[tid] += fabs(v[nx*iy + ix]);
    }

}

void jacobiStep(double *vp, double *v, double *f, int nx, int ny, double *e, double *w)
{
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 numberOfBlocks = dim3(8,8);
    jacobiStepKernel<<<numberOfBlocks, threadsPerBlock>>>(vp,v,f,nx,ny,e,w);
}

void weightBoundary_x(double *v, int nx, int ny, double *w, int iy)
{
    weightBoundaryKernel_x<<<8,32>>>(v,nx,ny,w,iy);
}

void weightBoundary_y(double *v, int nx, int ny, double *w, int ix)
{
    weightBoundaryKernel_x<<<8,32>>>(v,nx,ny,w,ix);
}

void sync()
{
    cudaDeviceSynchronize();
}

