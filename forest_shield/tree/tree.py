from typing import List, Tuple
import numpy as np
from numbers import Integral, Real
import copy
from scipy.sparse import issparse

from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
    BestFirstTreeBuilder,
    DepthFirstTreeBuilder,
    Tree,
)

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE


class DecisionTreeClassifier:
    def __init__(
        self,
        *,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str = "sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        random_state: int = None,
        class_weight: np.ndarray = None,
        ccp_alpha: float = 0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
    ):

        random_state = np.random.RandomState(self.random_state)

        n_samples, self.n_features_in_ = X.shape
        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        y = np.copy(y)

        if self.class_weight is not None:
            y_original = np.copy(y)

        y_encoded = np.zeros(y.shape, dtype=int)
        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes  # class encoding
        self.n_classes_ = classes.shape[0]  # number of classes
        y = y_encoded

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if self.class_weight is not None:
            # expanded_class_weight = compute_sample_weight(
            #     self.class_weight, y_original
            # )
            pass

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_samples_split = self.min_samples_split
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_classes_)
        else:
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
        splitter = self.splitter

        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )

        self.tree_ = Tree(self.n_features_in_, self.n_classes_)

        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )
        builder.build(self.tree_, X, y, sample_weight)

        return self
