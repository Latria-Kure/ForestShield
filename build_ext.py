#!/usr/bin/env python
"""
Build script for compiling Cython extensions.
"""
import os
import sys
import subprocess


def main():
    """Build the Cython extensions."""
    print("Building Cython extensions...")

    # Build in place
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

    try:
        subprocess.check_call(cmd)
        print("Build successful!")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
