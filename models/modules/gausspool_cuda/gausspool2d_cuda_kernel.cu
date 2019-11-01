#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

using namespace torch ;

#define MIN(a,b) (a) <= (b) ? (a) : (b)
#define MAX(a,b) (a) >= (b) ? (a) : (b)
#define THREADS 1024

namespace {

// Output shape of pooling
int pooling_output_shape_pad(int inputSize, int kernelSize, int pad, int stride, bool ceil_mode) 
{
  int outputSize = ((inputSize + pad + pad - (kernelSize - 1)
      - 1 + (ceil_mode ? stride - 1 : 0)) / stride + 1);
  if (pad) {
      // ensure that the last pooling starts inside the image
      // needed to avoid problems in ceil mode
      if ((outputSize - 1) * stride >= inputSize + pad)
        --outputSize;
  }
  return outputSize;
}

template <typename scalar_t>
__global__ void gauss_pool2d_forward_cuda_kernel(
    const scalar_t* const bottom_data, 
    const scalar_t* const param_data, 
    const int num, 
    const int channels,
    const int height, const int width, 
    const int pooled_height, const int pooled_width, 
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w,
    const int mode,
    const bool count_include_pad,
    scalar_t* const top_data)
{
  const int index = blockIdx.x * THREADS + threadIdx.x; //spatial dimension
  const int pw = index % pooled_width;
  const int ph = index / pooled_width;
  const int c = blockIdx.y ; //channel
  const int n = blockIdx.z ; //sample in batch
  
  //index in output
  const int oindex = (n * channels + c) * pooled_height * pooled_width + ph * pooled_width + pw ;

  if (ph < pooled_height) {
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = MIN(hstart + kernel_h, height + pad_h);
    int wend = MIN(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = MAX(hstart, 0);
    wstart = MAX(wstart, 0);
    hend = MIN(hend, height);
    wend = MIN(wend, width);
    if (!count_include_pad)
      pool_size = (hend - hstart) * (wend - wstart);

    // weight parameter
    int lm = 0 ;
    if( mode == 0 )      lm = 0 ;               //constant: [1]
    else if( mode == 1 ) lm = c  ;              //channel-wise: [1 x C x 1 x 1]
    else if( mode == 2 ) lm = n * channels + c ;//sample & channel-wise: [N x C x 1 x 1]
    else if( mode == 3 ) lm = oindex ;          //sample & channel & position-wise: [N x C x H x W]
    else if( mode == 4 ) lm = n ;               //sample-wise: [N x 1 x 1 x 1]
    scalar_t lambda = param_data[lm] ;

    // mean and standard deviation
    scalar_t aveval = 0 ;
    scalar_t stdval = 0 ;
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        scalar_t val = bottom_slice[h * width + w] ;
        aveval += val ;
        stdval += val * val ;
      }
    }
    aveval /= pool_size ;
    stdval = sqrt(MAX(stdval / pool_size - aveval * aveval, 1e-6)) ;

    top_data[oindex] = aveval + lambda * stdval ;
  }

}

template <typename scalar_t, bool OVERLAP>
__global__ void gauss_pool2d_backward_cuda_kernel(
    const scalar_t* const top_diff,
    const scalar_t* const bottom_data,
    const scalar_t* const param_data,
    const int num, 
    const int channels, 
    const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, 
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w,
    const int mode,
    const bool count_include_pad,
    scalar_t* const bottom_diff,
    scalar_t* const param_diff) 
{
  const int index = blockIdx.x * THREADS + threadIdx.x; //spatial dimension
  const int pw = index % pooled_width;
  const int ph = index / pooled_width;
  const int c = blockIdx.y ; //channel
  const int n = blockIdx.z ; //sample in batch

  //index in output
  const int oindex = (n * channels + c) * pooled_height * pooled_width + ph * pooled_width + pw ;

  if (ph < pooled_height) {
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = MIN(hstart + kernel_h, height + pad_h);
    int wend = MIN(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = MAX(hstart, 0);
    wstart = MAX(wstart, 0);
    hend = MIN(hend, height);
    wend = MIN(wend, width);
    if (!count_include_pad)
      pool_size = (hend - hstart) * (wend - wstart);

    // weight parameter
    int lm = 0 ;
    if( mode == 0 )      lm = 0 ;               //constant: [1]
    else if( mode == 1 ) lm = c  ;              //channel-wise: [1 x C x 1 x 1]
    else if( mode == 2 ) lm = n * channels + c ;//sample & channel-wise: [N x C x 1 x 1]
    else if( mode == 3 ) lm = oindex ;          //sample & channel & position-wise: [N x C x H x W]
    else if( mode == 4 ) lm = n ;               //sample-wise: [N x 1 x 1 x 1]
    scalar_t lambda = param_data[lm] ;
    scalar_t dtop  = top_diff[oindex] ;

    // mean and standard deviation
    scalar_t aveval = 0 ;
    scalar_t stdval = 0 ;
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        scalar_t val = bottom_slice[h * width + w] ;
        aveval += val ;
        stdval += val * val ;
      }
    }
    aveval /= pool_size ;
    stdval = sqrt(MAX(stdval / pool_size - aveval * aveval, 1e-6)) ;

    // derivatives
    scalar_t scl  = (dtop * lambda) / (pool_size * stdval) ;
    scalar_t bias = (dtop * (stdval - lambda * aveval)) / (pool_size * stdval) ;
    scalar_t* const bottom_diff_slice = bottom_diff + (n * channels + c) * height * width;
    // scalar_t scl = dtop / pool_size ;
    // scalar_t lambdaSigma = lambda / stdval ;
    if(OVERLAP){
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          scalar_t grad = scl * bottom_slice[h * width + w] + bias ;
          // scalar_t grad = scl * (1 + lambdaSigma * (bottom_slice[h * width + w] - aveval)) ;
          atomicAdd(bottom_diff_slice + (h * width + w), grad) ;
        }
      }
    }else{ // without overlap
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          scalar_t grad = scl * bottom_slice[h * width + w] + bias ;
          // scalar_t grad = scl * (1 + lambdaSigma * (bottom_slice[h * width + w] - aveval)) ;
          bottom_diff_slice[h * width + w] = grad ;
        }
      }
    }

    atomicAdd(param_diff + lm, dtop * stdval) ;
  }
}
} // namespace


// ---------------------------- //
void gauss_pool2d_forward_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) 
{
  // parameters
  AT_CHECK(
    ((kernel_size.size() == 1 || kernel_size.size() == 2) && (stride.size() == 1 || stride.size() == 2) && (padding.size() == 1 || padding.size() == 2)),
    "gauss_pool2d: all IntArrayRef sizes must be 2");

  AT_CHECK( (input.ndimension() == 3 || input.ndimension() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
  AT_CHECK( (param.ndimension() == 4), "non-empty 4D tensor expected for param");

  const int kH = kernel_size[0];
  const int kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  const int dH = stride.empty() ? kH : stride[0];
  const int dW = stride.empty() ? kW : stride[1];

  const int padH = padding[0];
  const int padW = padding.size() == 1 ? padH : padding[1];

  const int64_t nbatch      = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth  = input.size(-1);

  const int64_t outputWidth  = pooling_output_shape_pad(inputWidth, kW, padW, dW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape_pad(inputHeight, kH, padH, dH, ceil_mode);

  // weight parameter
  int mode = -1 ;
  if(      param.size(-4)==1      && param.size(-3)==1           && param.size(-2)==1            && param.size(-1)==1 )
    mode = 0 ; //constant: [1 x 1 x 1 x 1]
  else if( param.size(-4)==1      && param.size(-3)==nInputPlane && param.size(-2)==1            && param.size(-1)==1 )
    mode = 1 ; //channel-wise: [1 x C x 1 x 1]
  else if( param.size(-4)==nbatch && param.size(-3)==nInputPlane && param.size(-2)==1            && param.size(-1)==1 )
    mode = 2 ; //sample & channel-wise: [N x C x 1 x 1]
  else if( param.size(-4)==nbatch && param.size(-3)==nInputPlane && param.size(-2)==outputHeight && param.size(-1)==outputWidth )
    mode = 3 ; //sample & channel & position-wise: [N x C x H x W]
  else if( param.size(-4)==nbatch && param.size(-3)==1           && param.size(-2)==1            && param.size(-1)==1 )
    mode = 4 ; //sample-wise: [N x 1 x 1 x 1]
  else
    AT_CHECK(false, "param dimensions must be either [1 x 1 x 1 x 1], [1 x C x 1 x 1], [N x C x 1 x 1], [N x C x H x W] or [N x 1 x 1 x 1]");

  // init output tensor
  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  output.zero_();

  // CUDA kernel
  const int num_threads = THREADS ;
  const dim3 blocks((outputWidth*outputHeight + num_threads - 1) / num_threads, nInputPlane, nbatch);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gauss_pool2d_forward_cuda", 
    ([&] {
      scalar_t *output_data = output.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      scalar_t *param_data = param.data<scalar_t>();
      
      gauss_pool2d_forward_cuda_kernel<scalar_t>
      <<<blocks, num_threads>>>(
      input_data,
      param_data,
      nbatch,
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW,
      dH, dW,
      padH, padW,
      mode,
      count_include_pad,
      output_data);
    })
  );
  
  cudaError_t status = cudaGetLastError() ;
  AT_CHECK( (status == cudaSuccess), "gauss_pool2d_backward_cuda failed with error code") ;
 
  if (input.ndimension() == 3)
    output.resize_({nInputPlane, outputHeight, outputWidth});
}

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
    bool count_include_pad) 
{
  // parameters
  AT_CHECK(
    ((kernel_size.size() == 1 || kernel_size.size() == 2) && (stride.size() == 1 || stride.size() == 2) && (padding.size() == 1 || padding.size() == 2)),
    "gauss_pool2d: all IntArrayRef sizes must be 2");

  AT_CHECK( (input.ndimension() == 3 || input.ndimension() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
  AT_CHECK( (param.ndimension() == 4), "non-empty 4D tensor expected for param");

  const int kH = kernel_size[0];
  const int kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  const int dH = stride.empty() ? kH : stride[0];
  const int dW = stride.empty() ? kW : stride[1];

  const int padH = padding[0];
  const int padW = padding.size() == 1 ? padH : padding[1];

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  // output shape
  const int64_t outputWidth  = pooling_output_shape_pad(inputWidth, kW, padW, dW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape_pad(inputHeight, kH, padH, dH, ceil_mode);

  AT_CHECK( (gradOutput.size(-1) == outputWidth && gradOutput.size(-2) == outputHeight), "wrong size of gradOutput");
  
  // weight parameter
  int mode = -1 ;
  if(      param.size(-4)==1      && param.size(-3)==1           && param.size(-2)==1            && param.size(-1)==1 )
    mode = 0 ; //constant: [1 x 1 x 1 x 1]
  else if( param.size(-4)==1      && param.size(-3)==nInputPlane && param.size(-2)==1            && param.size(-1)==1 )
    mode = 1 ; //channel-wise: [1 x C x 1 x 1]
  else if( param.size(-4)==nbatch && param.size(-3)==nInputPlane && param.size(-2)==1            && param.size(-1)==1 )
    mode = 2 ; //sample & channel-wise: [N x C x 1 x 1]
  else if( param.size(-4)==nbatch && param.size(-3)==nInputPlane && param.size(-2)==outputHeight && param.size(-1)==outputWidth )
    mode = 3 ; //sample & channel & position-wise: [N x C x H x W]
  else if( param.size(-4)==nbatch && param.size(-3)==1           && param.size(-2)==1            && param.size(-1)==1 )
    mode = 4 ; //sample-wise: [N x 1 x 1 x 1]
  else
    AT_CHECK(false, "param dimensions must be either [1 x 1 x 1 x 1], [1 x C x 1 x 1], [N x C x 1 x 1], [N x C x H x W] or [N x 1 x 1 x 1]");

  // init derivative tensors
  gradInput.resize_as_(input);
  gradInput.zero_();
  gradParam.resize_as_(param);
  gradParam.zero_();

  // CUDA kernel
  const int num_threads = THREADS ;
  const dim3 blocks((outputWidth*outputHeight + num_threads - 1) / num_threads, nInputPlane, nbatch);

  if(dH < kH || dW < kW ){ // overlapped pooling
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
    "gauss_pool2d_backward_cuda", 
    [&] {
      scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
      scalar_t *gradInput_data = gradInput.data<scalar_t>();
      scalar_t *gradParam_data = gradParam.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      scalar_t *param_data = param.data<scalar_t>();

      gauss_pool2d_backward_cuda_kernel<scalar_t, true>
      <<<blocks, num_threads>>>(
          gradOutput_data,
          input_data,
          param_data,
          nbatch,
          nInputPlane,
          inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW,
          dH, dW,
          padH, padW,
          mode,
          count_include_pad,
          gradInput_data,
          gradParam_data);
      }
    );
  }else{
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
    "gauss_pool2d_backward_cuda", 
    [&] {
      scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
      scalar_t *gradInput_data = gradInput.data<scalar_t>();
      scalar_t *gradParam_data = gradParam.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      scalar_t *param_data = param.data<scalar_t>();

      gauss_pool2d_backward_cuda_kernel<scalar_t, false>
      <<<blocks, num_threads>>>(
          gradOutput_data,
          input_data,
          param_data,
          nbatch,
          nInputPlane,
          inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW,
          dH, dW,
          padH, padW,
          mode,
          count_include_pad,
          gradInput_data,
          gradParam_data);
      }
    );
  }
  
  cudaError_t status = cudaGetLastError() ;
  AT_CHECK( (status == cudaSuccess), "gauss_pool2d_backward_cuda failed with error code") ;
}
