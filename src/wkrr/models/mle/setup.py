import os

import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize


if os.name == "posix":
    extra_compile_args = ['-O3']
else:
    extra_compile_args = ['/O3']


ext_modules = [
    Extension(
        "_eval", 
        sources=["_eval.pyx"], 
        include_dirs=[np.get_include()], 
        libraries=["m"],  # Unix-like specific
        extra_compile_args=extra_compile_args
    )
]


setup(
    ext_modules=cythonize(ext_modules)
)
