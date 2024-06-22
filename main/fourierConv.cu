#include <cuFFT.h>
#include <cuBLAS.h>
#include <cuda.h>
#include <cudnn.h>
#include <torch/extension.h>

struct params {
    const unsigned int 
}
__host__ __forceinline__ struct define_kernel_parameters (void) {
    
}

template <typename T>
__global__ void conv2d_fwd_kernel (void) {
    

};


torch::Tensor conv2d_fwd (torch::Tensor[] input, torch::Tensor[] weight, torch::Tensor[] bias) {
    // from torch::Tensor to cudnn tensor
    cudnnTensorDescriptor_t in;
    cudnnTensorDescriptor_t w;
    cudnnTensorDescriptor_t b;

    in = cudnnGetTensor4dDescriptor();
    w = cudnnGetTensor4dDescriptor();
    b = cudnnGetTensor4dDescriptor();
    
    //Define the kernel based on the input

    // Run the kernel
    conv2d_fwd_kernel<<<gridSize, blockSize>>><float32>(&input, &weight, &bias);
};

__global__ void conv2d_bwd_kernel (void) {
    
}
