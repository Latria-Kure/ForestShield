from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "forest_shield.tree._criterion",
        ["forest_shield/tree/_criterion.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "forest_shield.tree._splitter",
        ["forest_shield/tree/_splitter.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "forest_shield.tree._tree",
        ["forest_shield/tree/_tree.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "forest_shield.tree._utils",
        ["forest_shield/tree/_utils.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "forest_shield.tree._partitioner",
        ["forest_shield/tree/_partitioner.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
]

setup(
    name="forest_shield",
    version="0.1.0",
    description="Forest Shield - A decision tree and random forest implementation",
    author="Forest Shield Team",
    packages=[
        "forest_shield",
        "forest_shield.tree",
        "forest_shield.forest",
        "forest_shield.utils",
    ],
    ext_modules=cythonize(extensions, language_level=3, gdb_debug=True),
    install_requires=[
        "numpy>=1.20.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",
)
