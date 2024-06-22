#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> conv2d_fwd(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
);

std::vector<torch::Tensor> conv2d_bwd(
    torch::Tensor grad_O,
    torch::Tensor O,
    torch::Tensor weights,
    torch::Tensor bias
);


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> conv2d(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return conv2d_fwd(input, weights, bias);
}

std::vector<torch::Tensor> conv2d_backward(
    torch::Tensor grad_O,
    torch::Tensor O,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(grad_O);
  CHECK_INPUT(O);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return conv2d_bwd(grad_o, O, weights, bias);
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d_fwd", &conv2d, "Convolution 2d forward (CUDA)");
  m.def("conv2d_bwd", &conv2d_backward, "Convolution 2d backward (CUDA)");
};