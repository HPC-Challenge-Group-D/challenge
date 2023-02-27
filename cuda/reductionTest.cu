/*
 * Simple test file for the reduction kernel
 */
#include <stdlib.h>
#include <stdio.h>

void deviceReduce(double *in, double* out, int N);
void deviceReduceMax(double *in, double* out, int N);

double referenceSum(double *in, int N)
{
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += in[i];
    }
    return sum;
}

double referenceMax(double *in, int N)
{
    double max = 0.0;
    for (int i = 0; i < N; i++)
    {
        max = in[i] > max ? in[i] : max;
    }
    return max;
}

int main(int argc, char *argv[])
{
    int N = 128;
    double *testArray;

    cudaMallocManaged(&testArray, N*sizeof(double));

    for(int i = 0; i < N; i++)
    {
        testArray[i] = i;
    }

    double *result;
    cudaMallocManaged(&result, 1*sizeof(double));

    deviceReduce(testArray,result,N);
    cudaDeviceSynchronize();

    double ref = referenceSum(testArray, N);
    printf("Reduction has yielded %f\n", result[0]);
    printf("Reference reduction has yielded %f\n", ref);
    if(fabs(ref - result[0]) < 1e-3)
        printf("Success\n");
    else
        printf("Failure\n");

    double *maxtest;
    cudaMallocManaged(&maxtest, 1* sizeof(double));


    deviceReduceMax(testArray, maxtest, N);
    cudaDeviceSynchronize();


    double maxref = referenceMax(testArray, N);

    printf("Max has yielded %f\n", maxtest[0]);
    printf("Reference max has yielded %f\n", maxref);
    if(fabs(maxref - maxtest[0]) < 1e-3)
        printf("Success\n");
    else
        printf("Failure\n");


    cudaFree(testArray);
    cudaFree(result);
    cudaFree(maxtest);
}