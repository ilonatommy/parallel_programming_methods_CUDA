 
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "helper_timer.h"

__global__ void add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
    }
}

void host_add(int *a, int *b, int *c, int N)
{
    int i;
    for(i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    long int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    
    cudaError_t err = cudaSuccess;
    
    int *dev_a, *dev_b, *dev_c;
    
    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *c = (int *)malloc(N * sizeof(int));
    int *ctrl_c = (int *)malloc(N * sizeof(int));
    
    //allocate memory - I added allocation success check
    err = cudaMalloc((void**)&dev_a,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&dev_b,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&dev_c,N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // initialise variables on the host
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    
    // copy the host input vectors in host memory to the device input vectors in device memory
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);
    
    //timer start:
    StopWatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
    // launch kernel, check if succeded
    //int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if(blocksPerGrid > 64) blocksPerGrid = 64;
    add <<<blocksPerGrid,threadsPerBlock>>> (dev_a,dev_b,dev_c, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //timer stop:
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float d_time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    // the same code in host version:
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);    
    host_add(a, b, ctrl_c, N);
    float h_time = sdkGetTimerValue(&timer);
    
    
    // Copy the device result vector in device memory to the host result vector in host memory.
    err = cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    
    // Verify that the result vector is correct
    for (int i = 0; i < N; i++)
    {
    	if (fabs(c[i] - ctrl_c[i] > 1e-5))
    	{
    		fprintf(stderr, "Result verification failed at element %d!\n", i);
        	exit(EXIT_FAILURE);
    	}
	}
    //printf("Test for %d elements: PASSED\n", N);    
       
    if(N<10)
    {
        for (int i = 0; i < N; i++) 
            printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    // print results:
    //printf ("Time for the device: %f ms, fot the host: %f ms.\n", d_time, h_time); 
    printf ("%ld, %d, %d, %f, %f.\n", N, blocksPerGrid, threadsPerBlock, d_time, h_time);

    return 0;
}
