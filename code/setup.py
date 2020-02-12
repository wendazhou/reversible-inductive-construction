import platform
import os
import subprocess
import sys

import setuptools
from setuptools import setup, find_packages, Extension

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

conda_prefix = os.getenv('CONDA_PREFIX')

if conda_prefix is not None and conda_prefix != '':
    rdkit_include_dirs = [conda_prefix + '/include/rdkit']
else:
    rdkit_include_dirs = []

torch_extensions = [
    CUDAExtension("genric.torch_extensions",
        sources=[os.path.join('cpp/torch/', fn) for fn in [
            'module.cpp',
            'repeat_interleave_cuda.cu',
            'repeat_interleave.cpp',
            'segment_logsumexp_backward.cu',
            'segment_logsumexp_cuda.cu',
            'segment_logsumexp.cpp',
            'segment_pool_cuda.cu',
            'segment_pool.cpp'
        ]],
        include_dirs=['lib/cub-1.8.0/'],
        extra_compile_args={
            'cxx': ['-g', '-DAT_PARALLEL_OPENMP', '-fopenmp'],
            'nvcc': []
        }),
]

extensions = [
    Extension("genric.genric_extensions.molecule_edit",
              sources=['cpp/molecule_edit.cpp'],
              include_dirs=rdkit_include_dirs,
              libraries=['boost_python', 'RDKitGraphMol', 'RDKitRDBoost']),
    Extension("genric.genric_extensions.molecule_representation",
              sources=['cpp/molecule_representation.cpp'],
              include_dirs=rdkit_include_dirs,
              libraries=['boost_python', 'RDKitGraphMol', 'RDKitRDBoost'])
]

class MyBuild(BuildExtension):
    def __init__(self, *args, **kwargs):
        super(MyBuild, self).__init__(*args, **kwargs)

    def build_extensions(self):
        self.extensions = torch_extensions
        super(MyBuild, self).build_extensions()

        self.extensions = extensions
        setuptools.command.build_ext.build_ext.build_extensions(self)



setup(
    name="induc-gen",
    packages=find_packages(),
    ext_modules=extensions + torch_extensions,
    cmdclass={
        'build_ext': MyBuild,
    }
)