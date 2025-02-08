#include <cuda/pipeline>
#include <stdio.h>

#define N 4

__global__ void asyncThreadScopeKernel(float *src, float *dst) {
    int tid = threadIdx.x;

    // Create a per-thread pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Async copy (each thread copies its own element)

    // 1. Reserve a spot for async work
    pipe.producer_acquire();
    // 2. Schedule the async memory copy
    cuda::memcpy_async(&dst[tid], &src[tid], sizeof(float), pipe);
    // 3. Finalize and launch async memory copy
    pipe.producer_commit();

    // 4. do something else while the memory is being copied 
    // !! something else
    printf("Thread %d doing something else...\n", tid);

    // 5. Wait for the data before using it.
    pipe.consumer_wait();
    // 6. Mark as complete and free pipeline resources.
    pipe.consumer_release();

    // Print result for debugging
    printf("Thread %d copied value: %f\n", tid, dst[tid]);
}

int main() {

    /*
    This demonstrates thread level async. 

    Here the kernel is simply to copy each value of the src array
    into the dst array. 
    
    h_src[i] is copied to h_dst[i] by thread i.




    
    
    */


    float h_src[N] = {1.0, 2.0, 3.0, 4.0};
    float h_dst[N] = {0.0};

    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);

    asyncThreadScopeKernel<<<1, N>>>(d_src, d_dst);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host result: %f %f %f %f\n", h_dst[0], h_dst[1], h_dst[2], h_dst[3]);

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
