"""
Setup script for Cython extensions.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "tree",
        ["tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
