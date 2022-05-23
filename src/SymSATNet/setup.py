import torch.cuda

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

gencode = [
        '-gencode=arch=compute_30,code=sm_30',
        '-gencode=arch=compute_35,code=sm_35',
        '-gencode=arch=compute_50,code=sm_50',
        '-gencode=arch=compute_52,code=sm_52',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_61,code=sm_61',
        '-gencode=arch=compute_61,code=compute_61',
]

if torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension(
            name = 'symsatnet._cuda',
            include_dirs = ['./src'],
            sources = [
                'src/symsatnet.cpp',
                'src/symsatnet_cuda.cu',
            ],
            extra_compile_args = {
                'cxx': ['-DMIX_USE_GPU', '-g'],
                'nvcc': ['-g', '-restrict', '-maxrregcount', '32', '-lineinfo', '-Xptxas=-v']
            }
        )
    ]

# Python interface
setup(
    name='symsatnet',
    version='0.0.1',
    install_requires=['torch>=1.3'],
    packages=['symsatnet'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
