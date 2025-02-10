#include <stdio.h>

#define N 2052

void __global__ reduce(const float* arr, int num_elems, float* out){

    int totalThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;    
    for(int i=idx; i <num_elems; i+= totalThreads){
        sum += arr[i];
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