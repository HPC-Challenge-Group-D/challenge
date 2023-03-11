//
// Created by Patrik Rac on 10.03.23.
//
#include "stdlib.h"
#include "stdio.h"
#include <mpi.h>

__global__
void demoKernel()
{
    printf("This is a demo kernel\n");
}


__global__
void initKernel(double *v, int N, double val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i+=stride)
        v[i] = val;
}

__global__
void addKernel(double *v, double *w, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i+=stride)
    {
        v[i] += w[i];
        printf("%f\n", v[i]);
    }

}


int main()
{
    /*Set the appropriate device before the call to MPI_init()*/
    char * localRankStr = NULL;
    int local_rank = 0;
    int deviceCount = 0;

    // We extract the local rank initialization using an environment variable
    if ((localRankStr = getenv("SLURM_LOCALID")) != NULL)
    {
        local_rank = atoi(localRankStr);
    }
    else
    {
        printf("Could not determine the appropriate local rank!\n");
    }
    cudaGetDeviceCount(&deviceCount);
    printf("There are %d devices\n", deviceCount);
    printf("Initializing with device %d\n", local_rank);
    fflush(stdout);
    cudaSetDevice(local_rank);

    /*Initialize MPI and the process info struct*/
    MPI_Init(NULL, NULL);
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
    {
        printf("Running MPI with CUDA support on %d procs\n", size);
        fflush(stdout);
    }

    demoKernel<<<1,4>>>();

    cudaDeviceSynchronize();


    double local_array[8];

    double *v, *w;

    cudaMalloc(&v, 8*sizeof(double));
    cudaMalloc(&w, 8*sizeof(double));

    initKernel<<<1,4>>>(v,8,10);

    cudaDeviceSynchronize();

    if(rank == 0)
    {
        MPI_Recv(w,8,MPI_DOUBLE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    else if(rank == 1)
    {
        MPI_Send(v,8,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }

    addKernel<<<1,8>>>(v,w,8);

    cudaDeviceSynchronize();

    cudaMemcpy(local_array, v, 8*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (rank == 0)
    {
        for(int i = 0; i < 8; i++)
            printf("%f\t",local_array[i]);
        printf("\n");
    }


    MPI_Finalize();


    cudaFree(v);
    cudaFree(w);
    return 0;
}