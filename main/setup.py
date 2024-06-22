from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourierConv2d',
    ext_modules=[
        CUDAExtension('fourierConv2d', [
            'fourierConv.cpp',
            'fourierConv.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
