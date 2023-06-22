from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

file_path = os.path.dirname(os.getcwd())
cutlass_path = "cutlass/cutlass/include"
file_path = os.path.join(file_path, cutlass_path)

setup(
    name='quantize_grad_weight_speed',
    ext_modules=[
        CUDAExtension(name='quantize_grad_weight_speed', sources=[
            'quantize_grad_weight_speed.cpp',
            'quantize_grad_weight_speed_kernel.cu'
        ], include_dirs=[file_path],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
