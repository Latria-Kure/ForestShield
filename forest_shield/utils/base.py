"""
Base classes for all estimators.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple, Callable


class BaseEstimator:
    """Base class for all estimators in forest_shield."""

    def __init__(self):
        self.is_fitted_ = False

    def _check_is_fitted(self):
        """Check if the estimator is fitted."""
        if not self.is_fitted_:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self.__dict__:
            if not key.endswith("_") and not key.startswith("_"):
                value = getattr(self, key)
                out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self

        for key, value in params.items():
            if key not in self.__dict__:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}."
                )
            setattr(self, key, value)
        return self


class ClassifierMixin:
    """Mixin class for all classifiers in forest_shield."""

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return np.mean(self.predict(X) == y)


class RegressorMixin:
    """Mixin class for all regressors in forest_shield."""

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v


class TransformerMixin:
    """Mixin class for all transformers in forest_shield."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y, **fit_params).transform(X)


class MetaEstimatorMixin:
    """Mixin class for all meta estimators in forest_shield."""

    def _check_estimator(self):
        """Check that the estimator is fitted."""
        if not hasattr(self, "estimator_"):
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this meta estimator."
            )
