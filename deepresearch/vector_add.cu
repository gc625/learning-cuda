#include <stdio.h>

#define N 1024

__global__ void vectorAdd(const float* a, const float* b, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n){return;}

    out[idx] = a[idx] + b[idx];


}

int main(){
    float a[N], b[N], c[N];

    for(int i = 0; i < N; ++i){
        a[i] = i;
        b[i] = i;
    }

    float (*d_A);
    float (*d_B);
    float (*d_C);


    cudaMalloc((void**)&d_A, N*sizeof(float));
    cudaMalloc((void**)&d_B, N*sizeof(float));
    cudaMalloc((void**)&d_C, N*sizeof(float));

    cudaMemcpy(d_A,&a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,&b,N*sizeof(float),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        // for N=512, 2 blocks each with 256? 
    printf("blocks %i\n",blocks);
    printf("threadsperblock %i\n",threadsPerBlock);
    vectorAdd<<<blocks,threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(&c,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i){
        // printf("gt: %f \n", a[i]+b[i]);
        // printf("gpu: %f \n",c[i]);
        if (a[i] + b[i] != c[i]){
            printf("erorr at idx: %i",i);
        }
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    

}