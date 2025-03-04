#!/usr/bin/env python
"""
Test script for the Criterion class.
"""
import sys
import pytest
import numpy as np

try:
    from forest_shield.tree._criterion import Criterion
except ImportError:
    print("Error: Could not import Criterion. Make sure to build the extension first.")
    print("Run: python build_ext.py")
    sys.exit(1)


def test_criterion_add():
    """Test the add method of the Criterion class."""
    criterion = Criterion()

    # Test with integers
    assert criterion.add(1, 2) == 3
    assert criterion.add(0, 0) == 0
    assert criterion.add(-1, 1) == 0

    # Test with floats
    assert criterion.add(1.5, 2.5) == 4.0
    assert abs(criterion.add(0.1, 0.2) - 0.3) < 1e-10  # Handle floating point precision

    # Test with large numbers
    assert criterion.add(1000000, 2000000) == 3000000

    print("All tests passed!")


if __name__ == "__main__":
    test_criterion_add()
