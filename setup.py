"""
Setup script for ForestShield.
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Cython extensions
extensions = [
    Extension(
        "forest_shield._cython.tree",
        ["forest_shield/_cython/tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
]

setup(
    name="forest_shield",
    version="0.1.0",
    description="A high-performance random forest implementation",
    author="ForestShield Team",
    author_email="info@forestshield.com",
    url="https://github.com/forestshield/forestshield",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "joblib>=1.0.0",
    ],
    setup_requires=[
        "cython>=0.29.0",
        "numpy>=1.20.0",
    ],
    ext_modules=cythonize(extensions, language_level=3),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
