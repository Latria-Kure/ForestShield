"""
Random Forest implementation.
"""

import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from ..utils.validation import check_array, check_X_y, check_random_state
from ..utils.parallel import parallel_build_trees, parallel_predict
from ..tree.tree import DecisionTreeClassifier
from ..utils.base import BaseEstimator, ClassifierMixin
import time


def _build_tree(X, y, sample_weight, random_state):
    """Build a single tree for the forest."""
    # Create a bootstrap sample
    n_samples = X.shape[0]
    random_state = check_random_state(random_state)
    indices = random_state.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]

    if sample_weight is not None:
        sample_weight_bootstrap = sample_weight[indices]
    else:
        sample_weight_bootstrap = None

    # Create and fit a tree
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=random_state,
    )
    tree.fit(X_bootstrap, y_bootstrap, sample_weight=sample_weight_bootstrap)

    return tree


def _predict_tree(tree, X):
    """Make predictions with a single tree."""
    probas = tree.predict_proba(X)
    # Ensure we have a 2D array for probabilities
    if len(probas.shape) == 1:
        # Convert to one-hot encoding if needed
        n_samples = X.shape[0]
        n_classes = len(tree.classes_)
        one_hot = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(probas):
            class_idx = np.where(tree.classes_ == pred)[0][0]
            one_hot[i, class_idx] = 1.0
        return one_hot
    return probas


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    criterion : str, default="gini"
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : int, float, or str, default="sqrt"
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          int(max_features * n_features) features are considered at each split.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
    n_jobs : int, default=None
        The number of jobs to run in parallel. None means 1.
    random_state : int, RandomState instance, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the sampling of the features to consider when
        looking for the best split at each node.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    """

    def __init__(
        self,
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.estimators_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : RandomForestClassifier
            Fitted estimator.
        """
        # Start timing the entire training process
        total_start_time = time.time()

        # Validate input
        X, y = check_X_y(X, y)

        # Ensure y is 1D
        y = y.ravel() if hasattr(y, "ravel") else y

        # Initialize
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape

        if self.verbose > 0:
            print(f"\n=== ForestShield RandomForestClassifier Training ===")
            print(f"Parameters:")
            print(f"  n_estimators: {self.n_estimators}")
            print(f"  max_depth: {self.max_depth}")
            print(f"  max_features: {self.max_features}")
            print(f"  criterion: {self.criterion}")
            print(f"  min_samples_split: {self.min_samples_split}")
            print(f"  min_samples_leaf: {self.min_samples_leaf}")
            print(f"  n_jobs: {self.n_jobs}")
            print(f"  bootstrap: {self.bootstrap}")
            print(f"Data:")
            print(f"  n_samples: {n_samples}")
            print(f"  n_features: {n_features}")
            print(f"  n_classes: {self.n_classes_}")
            print(f"  class distribution: {np.bincount(y.astype(int))}")

        # Generate random seeds for each tree
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        if self.verbose > 0:
            print(f"\nStarting training of {self.n_estimators} trees...")
            trees_start_time = time.time()

        # Build trees in parallel
        self.estimators_ = parallel_build_trees(
            _build_tree,
            self.n_estimators,
            X,
            y,
            sample_weight,
            seeds,
            self.n_jobs,
            self.verbose,
        )

        if self.verbose > 0:
            trees_time = time.time() - trees_start_time
            print(
                f"\nTraining completed: {len(self.estimators_)} trees built in {trees_time:.4f} seconds"
            )
            print(
                f"Average time per tree: {trees_time/max(1, len(self.estimators_)):.4f} seconds"
            )

        # Compute feature importances
        if self.verbose > 0:
            print("\nComputing feature importances...")

        self._compute_feature_importances(n_features)

        if self.verbose > 0:
            # Print top 5 feature importances
            indices = np.argsort(self.feature_importances_)[::-1]
            print("Top 5 features by importance:")
            for i in range(min(5, n_features)):
                print(
                    f"  Feature {indices[i]}: {self.feature_importances_[indices[i]]:.4f}"
                )

        # Set fitted flag
        self.is_fitted_ = True

        # Calculate and report total training time
        total_time = time.time() - total_start_time

        if self.verbose > 0:
            print(f"\n=== Training complete ===")
            print(f"Total training time: {total_time:.4f} seconds")

        return self

    def _compute_feature_importances(self, n_features):
        """Compute feature importances."""
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)

        # Sum feature importances from all trees
        for tree in self.estimators_:
            self.feature_importances_ += tree.feature_importances_

        # Normalize
        self.feature_importances_ /= len(self.estimators_)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fitted
        self._check_is_fitted()

        # Validate input
        X = check_array(X)

        # Predict in parallel
        proba = parallel_predict(
            _predict_tree, self.estimators_, X, self.n_jobs, self.verbose
        )

        # Ensure probabilities sum to 1
        if proba.sum(axis=1).max() > 1.0 + 1e-6:
            # Normalize if needed
            proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Check if we have any estimators
        if len(self.estimators_) == 0:
            print("Warning: No estimators available. Returning zeros.")
            return np.zeros(X.shape[0], dtype=int)

        proba = self.predict_proba(X)

        # Handle the case when proba is 1D (no classes found)
        if len(proba.shape) == 1:
            return np.zeros(proba.shape[0], dtype=int)

        return self.classes_[np.argmax(proba, axis=1)]
