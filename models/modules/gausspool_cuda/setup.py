from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gausspool2d_cuda',
    ext_modules=[
        CUDAExtension(
            name = 'gausspool2d_cuda', 
            sources = [ 'gausspool2d_cuda.cpp', 'gausspool2d_cuda_kernel.cu'],
            extra_compile_args={'cxx':[], 'nvcc': ['-arch=sm_60']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
