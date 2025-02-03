#include <stdio.h>
#include <assert.h>

#define N 4  // Matrix dimension N×N

// Kernel definition using 2D array syntax
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;  // x-coordinate
    int j = threadIdx.y;  // y-coordinate

    // Each thread adds one element of A and B
    // and writes to the corresponding element of C.
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // 1. Allocate host (CPU) memory as 2D arrays
    float h_A[N][N], h_B[N][N], h_C[N][N];

    // Initialize A and B with some data
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_A[i][j] = 1.0f;
            h_B[i][j] = 2.0f;
        }
    }

    // 2. Allocate device (GPU) memory as pointers to 2D arrays
    //    float (*d_A)[N] means "pointer to an array of N floats"
    float (*d_A)[N];
    float (*d_B)[N];
    float (*d_C)[N];

    // Each of these has N*N floats
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // 3. Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch the kernel
    //    One block of (N x N) threads
    dim3 threadsPerBlock(N, N);
    MatAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // 5. Copy the result (C) back to the host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Check the result
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            // Expect 1.0 + 2.0 = 3.0
            assert(h_C[i][j] == 3.0f);
        }
    }
    printf("Matrix add test passed!\n");

    // 7. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
