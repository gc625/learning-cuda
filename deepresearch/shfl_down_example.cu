#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp-level reduction using __shfl_down_sync.
// Each thread in a warp passes its value to a neighbor "delta" positions down.
// The mask 0xffffffff means all 32 threads are active.
__inline__ __device__ float warpReduceSum(float val) {
    // Loop: half the warp size each iteration.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // Each thread adds the value from the thread "offset" positions higher.
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Global kernel that reduces an array of floats into a single sum.
__global__ void reduceKernel(const float *input, float *output, int n) {
    // Compute a unique global thread index.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Each thread loads one element if within range.
    if (idx < n) {
        sum = input[idx];
    }
    
    // Use warp-level reduction: each warp computes a partial sum.
    sum = warpReduceSum(sum);
    
    // Only thread 0 in each warp writes the partial sum to global memory.
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        atomicAdd(output, sum);
    }
}

int main() {
    const int N = 1024;
    float h_input[N];
    // Initialize the host array. Here, we set each element to 1.0f so the expected sum is N.
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    float h_output = 0.0f;  // Will hold the reduction result on the host.
    
    // Allocate device memory.
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    // Copy host data to device.
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel.
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    
    // Copy the result back to host.
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("The sum is: %f (expected: %f)\n", h_output, (float)N);
    
    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
