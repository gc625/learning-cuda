#include <stdio.h>
#include <assert.h>
#include <random>

#define N 4 

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    // threadIdx is a predefined thread variable

    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // ----------------------------------------------------------------------
    // 1. Declare host (CPU) 2D arrays.
    //    Each is physically N*N floats in contiguous memory on the CPU.
    // ----------------------------------------------------------------------
    float host_A[N][N], host_B[N][N], host_C[N][N];

    // Fill host arrays with random data in [0,1].
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            // (float) cast ensures float division (rand() returns an int).
            host_A[i][j] = (float)rand() / (float)RAND_MAX;
            host_B[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }

    /*
      ------------------------------------------------------------------------
      2. Declare pointers for the device (GPU) memory.

      float (*device_A)[N];
      ---------------------
      "device_A is a pointer to an array of N floats." 
      That means we can use device_A[i][j] in a kernel as if it were A[i][j].
      But physically it's still a single contiguous block of memory sized N*N.

      Similarly for device_B and device_C.
      ------------------------------------------------------------------------
    */
    float (*device_A)[N];
    float (*device_B)[N];
    float (*device_C)[N];

    /*
      ------------------------------------------------------------------------
      3. cudaMalloc(void **devPtr, size_t size)
         - devPtr: pointer to a pointer to void. In other words, a parameter
                   where cudaMalloc will write back the GPU address it allocates.
         - size: how many bytes we want on the GPU.

         We pass &device_A (address of our pointer 'device_A') so cudaMalloc
         can store the allocated device address into 'device_A'.

         Because device_A is type "pointer to an array of N floats"
         (float (*)[N]), &device_A is thus "pointer to that pointer," or
         float (**)[N]. We cast it to (void**) to match the cudaMalloc signature.

         We need N*N*sizeof(float) bytes on the GPU for an N x N array of floats.
      ------------------------------------------------------------------------
    */
    cudaMalloc((void**)&device_A, N * N * sizeof(float));
    cudaMalloc((void**)&device_B, N * N * sizeof(float));
    cudaMalloc((void**)&device_C, N * N * sizeof(float));

    cudaMemcpy(device_A, host_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    //                  4 x 4 
    dim3 threadsPerBlock(N,N);

    MatAdd<<<1,threadsPerBlock>>>(device_A,device_B,device_C);
    //       ^ num blocks   
          
    // 4x4 matrix, launch 4x4=16 threads, each thread does one computation



    cudaDeviceSynchronize();
    cudaMemcpy(host_C, device_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    // 6. Check the result
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            // Expect 1.0 + 2.0 = 3.0
            float gt_c = host_A[i][j] + host_B[i][j];
            printf("ground truth c: %f, calculated c: %f\n", gt_c,host_C[i][j]);
            assert(fabs(host_C[i][j] - gt_c) < 1e-10);
        }
    }

    printf("Matrix add test passed!\n");

    // 7. Free device memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}
