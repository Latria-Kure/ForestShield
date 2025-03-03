"""
Feature selection using feature importances from a model.
"""

import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from ..utils.validation import check_array, check_X_y
from ..utils.base import BaseEstimator, TransformerMixin


def _get_feature_importances(estimator):
    """Get feature importances from estimator."""
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_

    if hasattr(estimator, "coef_"):
        if estimator.coef_.ndim == 1:
            return np.abs(estimator.coef_)
        else:
            return np.sum(np.abs(estimator.coef_), axis=0)

    raise ValueError(
        "The underlying estimator has no feature_importances_ or coef_ attribute."
    )


class SelectFromModel(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.
    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor.
        If True, ``estimator`` must be a fitted estimator.
        If False, ``estimator`` is fitted during the ``fit`` method.
    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.
    max_features : int, default=None
        The maximum number of features to select. If None, all features with
        importance greater than the threshold are kept.
    """

    def __init__(
        self, estimator, threshold=None, prefit=False, norm_order=1, max_features=None
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.feature_importances_ = None
        self.selected_features_ = None

    def _calculate_threshold(self, importances):
        """Calculate threshold based on feature importances."""
        if self.threshold is None:
            # Determine default threshold
            if hasattr(self.estimator, "penalty") and self.estimator.penalty == "l1":
                # L1-penalized models
                threshold = 1e-5
            else:
                threshold = "mean"
        else:
            threshold = self.threshold

        if isinstance(threshold, str):
            if "*" in threshold:
                scale, reference = threshold.split("*")
                scale = float(scale.strip())
                reference = reference.strip()

                if reference == "median":
                    reference_value = np.median(importances)
                elif reference == "mean":
                    reference_value = np.mean(importances)
                else:
                    raise ValueError(f"Unknown reference: {reference}")

                threshold = scale * reference_value

            elif threshold == "median":
                threshold = np.median(importances)

            elif threshold == "mean":
                threshold = np.mean(importances)

            else:
                raise ValueError(f"Unknown threshold: {threshold}")

        return float(threshold)

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values.
        **fit_params : dict
            Additional parameters to pass to the estimator.

        Returns
        -------
        self : SelectFromModel
            Fitted estimator.
        """
        # Validate input
        X = check_array(X)

        if not self.prefit:
            # Fit the estimator
            self.estimator.fit(X, y, **fit_params)

        # Get feature importances
        self.feature_importances_ = _get_feature_importances(self.estimator)

        # Calculate threshold
        threshold = self._calculate_threshold(self.feature_importances_)

        # Select features
        mask = self.feature_importances_ >= threshold

        # Apply max_features if specified
        if self.max_features is not None and np.sum(mask) > self.max_features:
            # Get indices of features sorted by importance
            indices = np.argsort(self.feature_importances_)[::-1]
            # Select top max_features
            mask = np.zeros_like(mask)
            mask[indices[: self.max_features]] = True

        self.selected_features_ = np.where(mask)[0]

        # Set fitted flag
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_r : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        # Check if fitted
        self._check_is_fitted()

        # Validate input
        X = check_array(X)

        # Select features
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : ndarray
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        # Check if fitted
        self._check_is_fitted()

        # Create mask
        mask = np.zeros(self.feature_importances_.shape, dtype=bool)
        mask[self.selected_features_] = True

        return self.selected_features_ if indices else mask
