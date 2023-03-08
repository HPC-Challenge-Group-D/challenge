//
// Created by Patrik Rac on 08.03.23.
//

#include "proc_info.h"
#include "init.h"

__global__
void initDataKernel(double *v, double *f, int nx, int ny, int offset_x, int offset_y)
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

void initDevice()
{
    /*Set the appropriate device before the call to MPI_init()*/
    char * localRankStr = NULL;
    int rank = 0, deviceCount = 0;

    // We extract the local rank initialization using an environment variable
    if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL)
    {
        rank = atoi(localRankStr);
    }
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount);
}

/*
 * Prepares the memory of v, vp and f on the device and initializes v and f with the correct values
 */
void prepareDataMemory(double *v, double *vp, double *f, int nx, int ny, int offset_x, int offset_y)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 numberOfBlocks = dim3(8,8);
    cudaMallocManaged(&v, nx * ny * sizeof(double));
    cudaMallocManaged(&f, nx * ny * sizeof(double));

    cudaMallocManaged(&vp, nx * ny * sizeof(double));

    /*Move the data used for the computation to the device*/
    cudaMemAdvise(v, nx*ny*sizeof(double), (cudaMemoryAdvise) 2, deviceId);
    cudaMemAdvise(vp, nx*ny*sizeof(double), (cudaMemoryAdvise) 2, deviceId);
    cudaMemAdvise(f, nx*ny*sizeof(double), (cudaMemoryAdvise) 1, deviceId);

    cudaMemPrefetchAsync(v, nx*ny*sizeof(double), deviceId);
    cudaMemPrefetchAsync(vp, nx*ny*sizeof(double), deviceId);
    cudaMemPrefetchAsync(f, nx*ny*sizeof(double), deviceId);

    initDataKernel<<<numberOfBlocks, threadsPerBlock>>>(v,f,nx,ny,offset_x, offset_y);
}

unsigned int prepareMiscMemory(double *w, double *e, double *w_device, double *e_device)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 numberOfBlocks = dim3(8,8);
    unsigned int num_gpu_threads = (numberOfBlocks.x * threadsPerBlock.x) * (numberOfBlocks.y * threadsPerBlock.y);
    cudaMallocManaged(&w_device, num_gpu_threads*sizeof(double));
    cudaMallocManaged(&e_device, num_gpu_threads*sizeof(double));

    cudaMemPrefetchAsync(w_device, num_gpu_threads*sizeof(double), deviceId);
    cudaMemPrefetchAsync(e_device, num_gpu_threads*sizeof(double), deviceId);

    cudaMallocManaged(&w, sizeof(double));
    cudaMallocManaged(&e, sizeof(double));
    return num_gpu_threads;
}

void freeDataMemory(double *v, double *vp, double *f)
{
    cudaFree(v);
    cudaFree(vp);
    cudaFree(f);
}


void freeMiscMemory(double *w, double *e, double *w_device, double *e_device)
{
    cudaFree(w);
    cudaFree(e);
    cudaFree(w_device);
    cudaFree(e_device);
}