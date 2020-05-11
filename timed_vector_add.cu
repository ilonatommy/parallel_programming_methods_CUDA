 
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

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
    int N = atoi(argv[1]);
    
    cudaError_t err = cudaSuccess;
    
    int a[N],b[N],c[N];
    int *dev_a, *dev_b, *dev_c;
    
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
    int blocksPerGrid = 1;
    int threadsPerBlock = N;
    add <<<blocksPerGrid,threadsPerBlock>>> (dev_a,dev_b,dev_c);
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
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);    
    host_add(a, b, c);    
    float h_time = sdkGetTimerValue(&timer);

    // Verify that the result vector is correct
    for (int i = 0; i < N; ++i)
    {
    	if (fabs(dev_c[i] - c[i] > 1e-5))
    	{
    		fprintf(stderr, "Result verification failed at element %d!\n", i);
        	exit(EXIT_FAILURE);
    	}
	}
    printf("Test PASSED\n");
    
    // Copy the device result vector in device memory to the host result vector in host memory.
    err = cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
        
    for (int i = 0; i < N; i++) {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    free(a);
    free(b);
    free(c);
    
    // print results:
    printf ("Time for the kernel: %f ms, fot the device: %f ms.\n", d_time, h_time);
    
    return 0;
}
