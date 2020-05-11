#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void d_add (int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a[tid]+b[tid];
    }
}

void h_add(int *a, int *b, int *c, int N)
{
    int i;
    for(i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void d_allocate_vector(int *v, int length)
{
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&v,length * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void h_verify_equal(int* v1, int* v2, int length)
{
    for (int i = 0; i < length; ++i)
    {
    	if (fabs(v1[i] - v2[i] > 1e-5))
    	{
    		fprintf(stderr, "Result verification failed at element %d!\n", i);
        	exit(EXIT_FAILURE);
    	}
	}
    printf("Test PASSED\n");
}

void h_check_kernel_errors()
{
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void h_copy2host(int* from, int* to, int length)
{
    err = cudaMemcpy(to, from, length*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    
    cudaError_t err = cudaSuccess;
    
    int a[N],b[N],c[N];
    int *dev_a, *dev_b, *dev_c;
    
    //allocate memory - I added allocation success check
    d_allocate_vector(dev_a, N);
    d_allocate_vector(dev_b, N);
    d_allocate_vector(dev_c, N);
    
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
    d_add <<<blocksPerGrid,threadsPerBlock>>> (dev_a,dev_b,dev_c);
    h_check_kernel_errors();
    
    //timer stop:
    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float d_time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    // the same code in host version:
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);    
    h_add(a, b, c);    
    float h_time = sdkGetTimerValue(&timer);

    // Verify that the result vector is correct
    h_verify_equal(dev_c, c, N);
    
    // Copy the device result vector in device memory to the host result vector in host memory.
    h_copy2host(c, dev_c, N);
        
    for (int i = 0; i < N; i++) printf("%d+%d=%d\n", a[i], b[i], c[i]);
    
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
