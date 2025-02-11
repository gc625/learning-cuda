#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void square_kernel(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float val = in[idx];
        out[idx] = val * val;
    }
}

torch::Tensor square_cuda(torch::Tensor input) {
    // Ensure input is on CUDA
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be Float32 tensor");
    auto output = torch::empty_like(input);  // allocate output tensor (same size/type)
    int N = input.numel();
    // Configure kernel launch (256 threads per block for example)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // Launch kernel (note: .data_ptr<float>() gives raw pointer to GPU memory)
    square_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    // It's good practice to synchronize or at least check for errors in C++ extension
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_cuda", &square_cuda, "Square elements of a tensor (CUDA)");
}