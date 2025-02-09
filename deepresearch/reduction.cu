#include <stdio.h>

#define N 1024

void __global__ reduce(const float* arr, int num_elems, float* out){

    int totalThreads = gridDim.x * blockDim.x;
    // grid = 1, blockdim = 256
    // total threads = 256

    int chunkSize = num_elems / totalThreads;

    // normally its
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // but blocksize = 1 -> blockIdx.x = 0 
    // int idx = threadIdx.x;

    float sum = 0;

    int startIdx = idx* chunkSize;
    
    for(int i=0; i <chunkSize; ++i){
        sum += arr[startIdx+i];
    }

    atomicAdd(out,sum);

}


int main(){
    float h_arr[N];
    float h_ret = 0;

    float gt_ret = 0;

    for(int i = 0; i < N;++i){
        h_arr[i] = i;
        gt_ret += i;
    }

    // create pointer to for gpu array 
    float *d_arr;
    float *d_ret;


    cudaMalloc((void**) &d_arr,sizeof(float)*N);
    cudaMalloc((void**) &d_ret,sizeof(float)); //assumes sum < max_float
    cudaMemcpy(d_arr,h_arr,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ret,&h_ret,sizeof(float),cudaMemcpyHostToDevice);

    int threadsPerBlock = N / 4;
    int blocks = 3;

    reduce<<<blocks,threadsPerBlock>>>(d_arr,N,d_ret);

    cudaMemcpy(&h_ret,d_ret,sizeof(float),cudaMemcpyDeviceToHost);

    printf("gt answer: %f\n",gt_ret);
    printf("reduce answer: %f\n",h_ret);

}