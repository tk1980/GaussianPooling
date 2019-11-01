from torch.utils.cpp_extension import load
gausspool2d_cuda = load(name='gausspool2d_cuda', sources=['gausspool2d_cuda.cpp', 'gausspool2d_cuda_kernel.cu'], verbose=True)
help(gausspool2d_cuda)
