"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Build a PyTorch extension that exposes the hand-written chunk_gla kernel.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gla_torch_ext',
    ext_modules=[
        CUDAExtension(
            name='gla_torch_ext',
            sources=[os.path.join(here, 'gla_torch_ext.cu')],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '-std=c++17',
                    '--use_fast_math',
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_86,code=sm_86',  # 30xx
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                ],
            },
            include_dirs=[here],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
