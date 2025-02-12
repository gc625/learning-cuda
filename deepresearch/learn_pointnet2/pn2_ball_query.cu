#include <cstdlib> 
#include <cmath>
#include <stdio.h>

void __global__ ball_query(const int b, const int n, const int m, const float r,
                    const float* xyz, const float* new_xyz, const int nsample, int* idx){

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y; // why? 

    if(pt_idx >m) return;

    // xyz = b * n * 3 array of original points
    // new_xyz = b * m * 3 array of query points
    // lets get x,y,z of curent query points


        // skip over prev batches
    new_xyz += batch_idx * m * 3 + pt_idx * 3;
    // move to the start of all points for this batch
    xyz += batch_idx * n * 3; 
    // move to the start of the result array where we will store the results
    // since ret = b,m,n_sample, same concept. Not *3 because this is just the idx
    idx += batch_idx * m * nsample + pt_idx * nsample;

    // init how many points we have successfully queried.
    int cnt = 0;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    float radius2 = r * r;


    for(int k = 0; k < n; ++k){

        float x = xyz[k*3 + 0];
        float y = xyz[k*3 + 1];
        float z = xyz[k*3 + 2];

        float dist2 = (new_x - x) * (new_x - x) +  (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);

        if(dist2 < radius2){
            
            if(cnt == 0){
                for(int i = 0; i < nsample; ++i){
                    idx[i] = k;
                }
            }

            idx[cnt] = k;
            ++cnt;
            if(cnt >= nsample) break;
        }

    }



}

void ball_query_cpu(int B, int N, int M, float radius, 
                    const float* xyz, const float* new_xyz, 
                    int nsample, int* idx_ref) 
{
    float radius2 = radius * radius;
    for (int b_i = 0; b_i < B; b_i++) {
        for (int m_i = 0; m_i < M; m_i++) {
            // Offsets for the CPU arrays:
            const float* new_ptr = new_xyz + (b_i * M + m_i) * 3;
            const float* xyz_ptr = xyz + b_i * N * 3;
            int* idx_ptr = idx_ref + (b_i * M + m_i) * nsample;

            float new_x = new_ptr[0];
            float new_y = new_ptr[1];
            float new_z = new_ptr[2];

            int cnt = 0;
            for (int k = 0; k < N; k++) {
                float x = xyz_ptr[k*3+0];
                float y = xyz_ptr[k*3+1];
                float z = xyz_ptr[k*3+2];
                float d2 = (new_x - x)*(new_x - x) 
                         + (new_y - y)*(new_y - y) 
                         + (new_z - z)*(new_z - z);
                if (d2 < radius2) {
                    // If first neighbor, fill everything with k
                    if (cnt == 0) {
                        for (int i = 0; i < nsample; i++) {
                            idx_ptr[i] = k;
                        }
                    }
                    idx_ptr[cnt] = k;
                    cnt++;
                    if (cnt >= nsample) break;
                }
            }
        }
    }
}



int main(){
    const int B = 16;
    const int N = 16384;  // number of points in the point cloud
    const int M = 4096;   // number of query points
    const int num_sample = 32;
    const float radius = 0.5;

    size_t xyz_size = B * N * 3 * sizeof(float);
    size_t new_xyz_size = B * M * 3 * sizeof(float);
    size_t idx_size = B * M * num_sample * sizeof(int);

    float* h_xyz = (float*) malloc(xyz_size);
    float* h_new_xyz = (float*) malloc(new_xyz_size);
    int*   h_idx = (int*) malloc(idx_size);  // If you want to copy results back later

    for (int i = 0; i < B * N * 3; ++i) {
        h_xyz[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < B * M * 3; ++i){
        h_new_xyz[i] = static_cast<float>(rand()) / RAND_MAX;
    }


    float* d_xyz;
    float* d_new_xyz;
    int* d_idx;

    cudaMalloc((void**)&d_xyz, xyz_size);
    cudaMalloc((void**)&d_new_xyz, new_xyz_size);
    cudaMalloc((void**)&d_idx, idx_size);
    
    cudaMemcpy(d_xyz,h_xyz,xyz_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_xyz,h_new_xyz,new_xyz_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,h_idx,idx_size,cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;

    dim3 blocks(ceil(M / threads_per_block),B);
    dim3 threads(threads_per_block);


    ball_query<<<blocks,threads>>>(
        B,N,M,radius,d_xyz,d_new_xyz,num_sample,d_idx
    );
    
    cudaMemcpy(h_idx,d_idx,idx_size,cudaMemcpyDeviceToHost);

    printf("first few idx, %i,%i,%i,%i\n",h_idx[245],h_idx[112],h_idx[22],h_idx[5]);


    int* h_idx_ref = (int*) malloc(idx_size);
    ball_query_cpu(B, N, M, radius, h_xyz, h_new_xyz, num_sample, h_idx_ref);
    
    bool all_match = true;
    for (size_t i = 0; i < (size_t) (B * M * num_sample); i++) {
        if (h_idx_ref[i] != h_idx[i]) {
            printf("Mismatch at i=%zu: ref=%d, gpu=%d\n", i, h_idx_ref[i], h_idx[i]);
            all_match = false;
            break;
        }
    }

    if (all_match) {
        printf("CPU and GPU results match!\n");
    }

}

