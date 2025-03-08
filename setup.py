from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import os
import sys

# Check if Cython is installed
try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    if not os.path.exists(os.path.join("forest_shield", "tree", "_tree.c")):
        print("Cython is required to build the extension modules.")
        sys.exit(1)

# Define the base set of compiler flags
# This setup is suitable for building the package for development
extra_compile_args = ["-O3", "-march=native", "-ffast-math"]
if sys.platform == "darwin":
    extra_compile_args.extend(["-stdlib=libc++", "-mmacosx-version-min=10.9"])
elif sys.platform == "win32":
    extra_compile_args = ["/O3", "/arch:AVX2"]

# Define the extensions
extensions = [
    Extension(
        "forest_shield.tree._tree",
        ["forest_shield/tree/_tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "forest_shield.tree._splitter",
        ["forest_shield/tree/_splitter.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "forest_shield.tree._criterion",
        ["forest_shield/tree/_criterion.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "forest_shield.tree._utils",
        ["forest_shield/tree/_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "forest_shield.tree._partitioner",
        ["forest_shield/tree/_partitioner.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    ),
]

# Apply cythonize if available
if USE_CYTHON:
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )


# Custom build_ext command to handle compiler flags
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Detect compiler type
        compiler = self.compiler.compiler_type

        # Adjust compiler flags based on compiler type
        if compiler == "msvc":
            for e in self.extensions:
                e.extra_compile_args = ["/O3", "/arch:AVX2"]
        else:
            # For gcc and clang
            for e in self.extensions:
                if e.name == "forest_shield.tree._tree":
                    e.extra_compile_args.append("-std=c++14")

        build_ext.build_extensions(self)


setup(
    name="forest_shield",
    version="0.1.0",
    description="Forest Shield - A decision tree and random forest implementation",
    author="Forest Shield Team",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={"build_ext": CustomBuildExt},
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "scikit-learn>=1.0.0",  # Added for test dependencies
    ],
    python_requires=">=3.7",
    # Development mode installation is supported
    zip_safe=False,
)
