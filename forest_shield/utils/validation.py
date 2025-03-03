"""
Utilities for input validation and data processing.
"""

import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Callable


def check_array(
    X,
    dtype=None,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    copy=True,
):
    """Input validation on an array, list, or similar.

    Parameters
    ----------
    X : array-like
        Input object to check / convert.
    dtype : dtype, default=None
        Data type to force. If None, then the type will be determined from X.
    ensure_2d : bool, default=True
        Whether to make X at least 2d.
    allow_nd : bool, default=False
        Whether to allow X to be a n-dimensional array.
    ensure_min_samples : int, default=1
        Make sure that X has at least this number of samples.
    ensure_min_features : int, default=1
        Make sure that X has at least this number of features.
    copy : bool, default=True
        Whether a forced copy will be triggered.

    Returns
    -------
    X_converted : ndarray
        The converted and validated X.
    """
    if isinstance(X, np.ndarray):
        # Handle numpy array
        if copy and np.may_share_memory(X, X):
            X = np.array(X, dtype=dtype)
    else:
        # Convert to numpy array
        X = np.array(X, dtype=dtype)

    if ensure_2d:
        # Ensure at least 2 dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 0:
            X = X.reshape(1, 1)

    if not allow_nd and X.ndim > 2:
        raise ValueError(f"Found array with dim {X.ndim}. Expected <= 2.")

    if ensure_min_samples > 0 and X.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {X.shape[0]} sample(s) but a minimum of "
            f"{ensure_min_samples} is required."
        )

    if ensure_min_features > 0 and X.ndim > 1 and X.shape[1] < ensure_min_features:
        raise ValueError(
            f"Found array with {X.shape[1]} feature(s) but a minimum of "
            f"{ensure_min_features} is required."
        )

    return X


def check_X_y(
    X,
    y,
    dtype=None,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
):
    """Input validation for standard estimators.

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Target values.
    dtype : dtype, default=None
        Data type to force.
    ensure_2d : bool, default=True
        Whether to make X at least 2d.
    allow_nd : bool, default=False
        Whether to allow X to be a n-dimensional array.
    multi_output : bool, default=False
        Whether to allow 2-d y (n_samples, n_outputs).
    ensure_min_samples : int, default=1
        Make sure that X and y have at least this number of samples.
    ensure_min_features : int, default=1
        Make sure that X has at least this number of features.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type.

    Returns
    -------
    X_converted : ndarray
        The converted and validated X.
    y_converted : ndarray
        The converted and validated y.
    """
    X = check_array(
        X,
        dtype=dtype,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
    )

    if y is None:
        return X, y

    y = np.array(y, dtype=dtype)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if not multi_output and y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(f"y has {y.shape[1]} dimensions, but multi_output is False.")

    if y_numeric and not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"y has non-numeric dtype: {y.dtype}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
        )

    return X, y


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, or RandomState instance
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.

    Returns
    -------
    random_state : RandomState instance
        The random state object based on seed.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a numpy.random.RandomState instance"
    )
