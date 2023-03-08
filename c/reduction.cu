/*
Source file handling the reduction on a single device.
*/
#include "reduction.h"

#define FULL_MASK 0xffffffff

__inline__ __device__
double warpReduceSum(double val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__inline__ __device__
double warpReduceMax(double val) {
    double tmp;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        tmp = __shfl_down_sync(FULL_MASK, val, offset);
        val = val > tmp ? val : tmp;
    }

    return val;
}


__inline__ __device__
double blockReduceSum(double val) {

    static __shared__ double shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__inline__ __device__
double blockReduceMax(double val) {

    static __shared__ double shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

    return val;
}

__global__ void deviceReduceKernel(double *in, double* out, int N) {
    double sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x]=sum;
}

__global__ void deviceReduceKernelMax(double *in, double* out, int N) {
    double m = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        m = in[i] > m ? in[i] : m;
    }
    m = blockReduceMax(m);
    if (threadIdx.x==0)
        out[blockIdx.x]=m;
}

void deviceReduce(double *in, double* out, int N) {
    int threads = 512;
    int blocks = min((N + threads - 1) / threads, 1024);

    deviceReduceKernel<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

void deviceReduceMax(double *in, double* out, int N) {
    int threads = 512;
    int blocks = min((N + threads - 1) / threads, 1024);

    deviceReduceKernelMax<<<blocks, threads>>>(in, out, N);
    deviceReduceKernelMax<<<1, 1024>>>(out, out, blocks);
}





