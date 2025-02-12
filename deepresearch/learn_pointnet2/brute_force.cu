// knn_example.cu
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Helper macro to check CUDA errors
#define CHECK_CUDA(call) {                                      \
    cudaError_t err = call;                                     \
    if(err != cudaSuccess) {                                    \
        std::cerr << "CUDA error in " << __FILE__ << ":"        \
                  << __LINE__ << ": " << cudaGetErrorString(err) \
                  << std::endl;                                 \
        exit(1);                                              \
    }                                                         \
}

// ------------------------------------------------------------------
// Kernel: Brute-force KNN search
//
// For each query point (indexed by q_i), the kernel scans all points 
// (indexed by j) and maintains an array of the best (smallest) K distances.
// Note: For simplicity, this example assumes the following:
//   - Points are 3D (dim == 3)
//   - K (number of neighbors) is at most 32
// ------------------------------------------------------------------
__global__ void knn_bruteforce_kernel(const float* __restrict__ points,
                                      const float* __restrict__ queries,
                                      int* __restrict__ knn_idx,
                                      float* __restrict__ knn_dist,
                                      int N, int M, int dim, int K) {
    // Each thread handles one query point.
    int q_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_i >= M) return;

    // Fixed-size arrays for K best distances and indices.
    // (We assume K <= 32; if you need a larger K, adjust accordingly.)
    float best_dist[32];
    int best_idx[32];

    // Initialize the best distances to a very large value and indices to -1.
    for (int k = 0; k < K; ++k) {
        best_dist[k] = 1e10f;
        best_idx[k] = -1;
    }

    // Copy the query point from global memory into a local array.
    // Since we assume dim == 3, we declare q[3].
    float q[3];
    for (int d = 0; d < dim; ++d) {
        q[d] = queries[q_i * dim + d];
    }

    // Loop over all points to compute distances.
    for (int j = 0; j < N; ++j) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = q[d] - points[j * dim + d];
            dist2 += diff * diff;
        }

        // If this distance is smaller than the worst in our current list,
        // find the worst candidate and replace it.
        if (dist2 < best_dist[0]) {
            int max_k = 0;
            for (int k = 1; k < K; ++k) {
                if (best_dist[k] > best_dist[max_k])
                    max_k = k;
            }
            if (dist2 < best_dist[max_k]) {
                best_dist[max_k] = dist2;
                best_idx[max_k] = j;
            }
        }
    }

    // Write the final K nearest neighbor indices and distances to global memory.
    for (int k = 0; k < K; ++k) {
        knn_idx[q_i * K + k] = best_idx[k];
        knn_dist[q_i * K + k] = best_dist[k];
    }
}





__global__ void ball_query_bruteforce_kernel(const float* __restrict__ points,
                                        const float* __restrict__ queries,
                                        int* __restrict__ ball_idx,
                                        float* __restrict__ ball_dist,
                                        int N, int M, int dim, float r, int max_neighours) {
    // Each thread handles one query point.
    int q_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(q_i >= M) return;

    int count = 0;

    // Copy the query point from global memory into a local array.
    // Since we assume dim == 3, we declare q[3].
    float q[3];
    for (int d = 0; d < dim; ++d) {
        q[d] = queries[q_i * dim + d];
    }

    // Loop over all points to compute distances.
    for (int j = 0; j < N; ++j) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = q[d] - points[j * dim + d];
            dist2 += diff * diff;
        }

        // If this distance is smaller than the worst in our current list,
        // find the worst candidate and replace it.
        if (dist2 < r * r) {
            if (count < max_neighours) {
                // "Append" this point's index and distance by writing them into the output array.
                ball_idx[q_i * max_neighours + count] = j;
                ball_dist[q_i * max_neighours + count] = sqrt(dist2);
                count++;
            }
        }
    }
}






// ------------------------------------------------------------------
// Main function: Setup data, launch the kernel, and print results
// ------------------------------------------------------------------
int test_knn() {
    // Parameters
    const int N = 10000000;  // number of points in the point cloud
    const int M = 10000;   // number of query points
    const int dim = 3;   // dimension of each point (3D)
    const int K = 100;     // number of nearest neighbors to find

    size_t points_size = N * dim * sizeof(float);
    size_t queries_size = M * dim * sizeof(float);
    size_t knn_idx_size = M * K * sizeof(int);
    size_t knn_dist_size = M * K * sizeof(float);

    // Allocate host memory
    float* h_points   = (float*) malloc(points_size);
    float* h_queries  = (float*) malloc(queries_size);
    int*   h_knn_idx  = (int*)   malloc(knn_idx_size);
    float* h_knn_dist = (float*) malloc(knn_dist_size);

    // Initialize points and queries with random data in [0, 1)
    for (int i = 0; i < N * dim; i++) {
        h_points[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * dim; i++) {
        h_queries[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_points;   float* d_queries;
    int*   d_knn_idx;  float* d_knn_dist;
    CHECK_CUDA(cudaMalloc((void**)&d_points, points_size));
    CHECK_CUDA(cudaMalloc((void**)&d_queries, queries_size));
    CHECK_CUDA(cudaMalloc((void**)&d_knn_idx, knn_idx_size));
    CHECK_CUDA(cudaMalloc((void**)&d_knn_dist, knn_dist_size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries, queries_size, cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    
    // Launch the kernel
    knn_bruteforce_kernel<<<blocks, threads>>>(d_points, d_queries, d_knn_idx, d_knn_dist, N, M, dim, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the results from device to host
    CHECK_CUDA(cudaMemcpy(h_knn_idx, d_knn_idx, knn_idx_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_knn_dist, d_knn_dist, knn_dist_size, cudaMemcpyDeviceToHost));

    // Print the nearest neighbors for the first query point (query 0)
    std::cout << "Query 0's " << K << " nearest neighbors:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Neighbor " << k 
                  << ": index = " << h_knn_idx[k] 
                  << ", distance^2 = " << h_knn_dist[k] << std::endl;
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_knn_idx);
    cudaFree(d_knn_dist);

    // Free host memory
    free(h_points);
    free(h_queries);
    free(h_knn_idx);
    free(h_knn_dist);

    return 0;
}

int test_ball_query() {
    // Parameters
    const int N = 10000;  // number of points in the point cloud
    const int M = 100;   // number of query points
    const int dim = 3;   // dimension of each point (3D)
    const int K = 32;     // number of nearest neighbors to find
    const float R = 0.2;
    size_t points_size = N * dim * sizeof(float);
    size_t queries_size = M * dim * sizeof(float);
    size_t knn_idx_size = M * K * sizeof(int);
    size_t knn_dist_size = M * K * sizeof(float);

    // Allocate host memory
    float* h_points   = (float*) malloc(points_size);
    float* h_queries  = (float*) malloc(queries_size);
    int*   h_knn_idx  = (int*)   malloc(knn_idx_size);
    float* h_knn_dist = (float*) malloc(knn_dist_size);

    // Initialize points and queries with random data in [0, 1)
    for (int i = 0; i < N * dim; i++) {
        h_points[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * dim; i++) {
        h_queries[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_points;   float* d_queries;
    int*   d_knn_idx;  float* d_knn_dist;
    CHECK_CUDA(cudaMalloc((void**)&d_points, points_size));
    CHECK_CUDA(cudaMalloc((void**)&d_queries, queries_size));
    CHECK_CUDA(cudaMalloc((void**)&d_knn_idx, knn_idx_size));
    CHECK_CUDA(cudaMalloc((void**)&d_knn_dist, knn_dist_size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries, queries_size, cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    
    // Launch the kernel
    ball_query_bruteforce_kernel<<<blocks, threads>>>(d_points, d_queries, d_knn_idx, d_knn_dist, N, M, dim,R, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the results from device to host
    CHECK_CUDA(cudaMemcpy(h_knn_idx, d_knn_idx, knn_idx_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_knn_dist, d_knn_dist, knn_dist_size, cudaMemcpyDeviceToHost));

    // Print the nearest neighbors for the first query point (query 0)
    std::cout << "Query 0's " << K << " nearest neighbors:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Neighbor " << k 
                  << ": index = " << h_knn_idx[k] 
                  << ", distance = " << h_knn_dist[k] << std::endl;
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_knn_idx);
    cudaFree(d_knn_dist);

    // Free host memory
    free(h_points);
    free(h_queries);
    free(h_knn_idx);
    free(h_knn_dist);

    return 0;
}


int main(){
    test_ball_query();
}