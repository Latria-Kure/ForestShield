"""
Decision Tree implementation.
"""

import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from ..utils.validation import check_array, check_X_y, check_random_state


class Node:
    """A node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted value for leaf nodes

    @property
    def node_count(self):
        """Return the number of nodes in the subtree rooted at this node."""
        count = 1  # Count this node
        if self.left is not None:
            count += self.left.node_count
        if self.right is not None:
            count += self.right.node_count
        return count


class DecisionTree:
    """Base class for decision trees.

    Parameters
    ----------
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
    max_features : int, float, or str, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          int(max_features * n_features) features are considered at each split.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def _calculate_impurity(self, y):
        """Calculate impurity of a node."""
        m = len(y)
        if m == 0:
            return 0

        # Count classes
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m

        if self.criterion == "gini":
            # Gini impurity
            return 1 - np.sum(probabilities**2)
        else:
            # Entropy
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _calculate_information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split."""
        p = len(y_left) / len(y)
        return (
            self._calculate_impurity(y)
            - p * self._calculate_impurity(y_left)
            - (1 - p) * self._calculate_impurity(y_right)
        )

    def _best_split(self, X, y, features):
        """Find the best split for a node."""
        m, n = X.shape
        if m <= 1:
            return None, None

        # Count classes in current node
        parent_impurity = self._calculate_impurity(y)

        # Initialize variables
        best_info_gain = -float("inf")
        best_feature = None
        best_threshold = None

        # Try each feature
        for feature in features:
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature])

            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                # Skip if split doesn't meet min_samples_leaf
                if (
                    np.sum(left_indices) < self.min_samples_leaf
                    or np.sum(right_indices) < self.min_samples_leaf
                ):
                    continue

                # Calculate information gain
                y_left = y[left_indices]
                y_right = y[right_indices]
                info_gain = self._calculate_information_gain(y, y_left, y_right)

                # Update best split
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        m, n = X.shape

        # Check if we should stop splitting
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or m < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # Create a leaf node
            # Ensure y is 1D before using bincount
            y_flat = y.ravel() if hasattr(y, "ravel") else y
            value = np.bincount(y_flat.astype(int), minlength=self.n_classes_)
            value = value / m  # Convert to probabilities
            return Node(value=value)

        # Determine features to consider
        if self.max_features == "sqrt":
            n_features = int(np.sqrt(n))
        elif self.max_features == "log2":
            n_features = int(np.log2(n))
        elif isinstance(self.max_features, int):
            n_features = min(self.max_features, n)
        elif isinstance(self.max_features, float):
            n_features = int(self.max_features * n)
        else:
            n_features = n

        # Randomly select features
        random_state = check_random_state(self.random_state)
        features = random_state.choice(n, size=n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, features)

        # If no valid split found, create a leaf node
        if best_feature is None:
            # Ensure y is 1D before using bincount
            y_flat = y.ravel() if hasattr(y, "ravel") else y
            value = np.bincount(y_flat.astype(int), minlength=self.n_classes_)
            value = value / m  # Convert to probabilities
            return Node(value=value)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Create a decision node
        node = Node(
            feature=best_feature,
            threshold=best_threshold,
            left=self._build_tree(X_left, y_left, depth + 1),
            right=self._build_tree(X_right, y_right, depth + 1),
        )

        return node

    def _compute_feature_importances(self, X, y):
        """Compute feature importances."""
        self.feature_importances_ = np.zeros(self.n_features_)
        self._update_feature_importances(self.tree_, X, y, self.feature_importances_)

        # Normalize
        normalizer = np.sum(self.feature_importances_)
        if normalizer > 0:
            self.feature_importances_ /= normalizer

    def _update_feature_importances(self, node, X, y, importances, weighted=True):
        """Update feature importances recursively."""
        if node.feature is None:  # Leaf node
            return

        # Split the data
        left_indices = X[:, node.feature] <= node.threshold
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Calculate the improvement in impurity
        parent_impurity = self._calculate_impurity(y)
        left_impurity = self._calculate_impurity(y_left)
        right_impurity = self._calculate_impurity(y_right)

        # Weight by the number of samples
        n_samples = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if weighted:
            weight = n_samples
        else:
            weight = 1

        # Update feature importance
        importances[node.feature] += weight * (
            parent_impurity
            - (n_left / n_samples) * left_impurity
            - (n_right / n_samples) * right_impurity
        )

        # Recursively update for child nodes
        self._update_feature_importances(
            node.left, X_left, y_left, importances, weighted
        )
        self._update_feature_importances(
            node.right, X_right, y_right, importances, weighted
        )

    def fit(self, X, y, sample_weight=None):
        """Build a decision tree classifier from the training set (X, y).

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
        self : DecisionTree
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)

        # Ensure y is 1D
        y = y.ravel() if hasattr(y, "ravel") else y

        # Initialize
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            X_weighted = np.repeat(
                X,
                np.round(
                    sample_weight * len(sample_weight) / np.sum(sample_weight)
                ).astype(int),
                axis=0,
            )
            y_weighted = np.repeat(
                y,
                np.round(
                    sample_weight * len(sample_weight) / np.sum(sample_weight)
                ).astype(int),
                axis=0,
            )
            X, y = X_weighted, y_weighted

        # Build the tree
        self.tree_ = self._build_tree(X, y)

        # Compute feature importances
        self._compute_feature_importances(X, y)

        # Set fitted flag
        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = check_array(X)

        # Check if fitted
        if self.tree_ is None:
            raise ValueError("Tree not fitted. Call fit before predict_proba.")

        # Predict for each sample
        return np.array([self._predict_proba_sample(x, self.tree_) for x in X])

    def _predict_proba_sample(self, x, node):
        """Predict class probabilities for a single sample."""
        if node.value is not None:  # Leaf node
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)

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
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class DecisionTreeClassifier(DecisionTree):
    """A decision tree classifier.

    Parameters
    ----------
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
    max_features : int, float, or str, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          int(max_features * n_features) features are considered at each split.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        depth : int
            The maximum depth of the tree.
        """
        if not hasattr(self, "tree_"):
            return 0

        def _get_node_depth(node, current_depth=0):
            if node is None:
                return current_depth
            if node.left is None and node.right is None:
                return current_depth
            return max(
                _get_node_depth(node.left, current_depth + 1),
                _get_node_depth(node.right, current_depth + 1),
            )

        return _get_node_depth(self.tree_)
