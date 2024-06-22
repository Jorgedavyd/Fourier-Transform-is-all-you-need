#include <cuFFT.h>
#include <cuBLAS.h>
#include <cuda.h>
#include <cudnn.h>
#include <torch/extension.h>

struct params {
    const unsigned int i,
}
__host__ __forceinline__ struct define_kernel_parameters (void) {
    
}

template <typename T>
__global__ void conv2d_fwd_kernel (
    cudnnTensorDescriptor_t<T> input,
    cudnnTensorDescriptor_t<T> weight,
    cudnnTensorDescriptor_t<T> bias,

) {

};


torch::Tensor conv2d_fwd (torch::Tensor[] input, torch::Tensor[] weight, torch::Tensor[] bias) {
    // cudnn handler
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // from torch::Tensor to cudnn tensor
    cudnnTensorDescriptor_t in;
    cudnnTensorDescriptor_t w;
    cudnnTensorDescriptor_t b;
    cudnnTensorDescriptor_t out;

    cudnnGetTensor4dDescriptor(&in);
    cudnnGetTensor4dDescriptor(&w);
    cudnnGetTensor4dDescriptor(&b);
    cudnnGetTensor4dDescriptor(&out);
    
    //Define the kernel based on the input

    // Run the kernel
    conv2d_fwd_kernel<<<gridSize, blockSize>>><float32>(&input, &weight, &bias);
    
    cudnnDestroyTensorDescriptor(in);
    cudnnDestroyTensorDescriptor(w);
    cudnnDestroyTensorDescriptor(b);

    cudnnDestroy(cudnn);

    return torch::from_blob(out);
};

__global__ void conv2d_bwd_kernel (void) {
    
}
