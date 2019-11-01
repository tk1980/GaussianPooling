#include <torch/extension.h>

#include <vector>

// CUDA kernel function declarations

void gauss_pool2d_forward_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) ;

void gauss_pool2d_backward_cuda(
    torch::Tensor& gradInput,
    torch::Tensor& gradParam,
    const torch::Tensor& gradOutput,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) ;

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gauss_pool2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad)
{
  CHECK_INPUT(input) ;
  CHECK_INPUT(param) ;
  torch::Tensor output = at::empty({0}, input.options()) ;

  gauss_pool2d_forward_cuda(output, input, param, kernel_size, stride, padding, ceil_mode, count_include_pad) ;

  return output ;
}

std::vector<torch::Tensor> gauss_pool2d_backward(
    const torch::Tensor& gradOutput,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad)
{
  CHECK_INPUT(gradOutput) ;
  CHECK_INPUT(input) ;
  CHECK_INPUT(param) ;
  
  torch::Tensor gradInput = at::empty({0}, input.options()) ;
  torch::Tensor gradParam = at::empty({0}, param.options()) ;

  gauss_pool2d_backward_cuda(gradInput, gradParam, gradOutput, input, param, kernel_size, stride, padding, ceil_mode, count_include_pad) ;

  return {gradInput, gradParam} ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gauss_pool2d_forward, "Gaussian Pooling forward (CUDA)") ;
  m.def("backward", &gauss_pool2d_backward, "Gaussian Pooling backward (CUDA)") ;
}
