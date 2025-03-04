from typing import List, Tuple
import numpy as np
from numbers import Integral, Real


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str = "sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        random_state: int = None,
        ccp_alpha: float = 0.0,
        monotonic_cst: List[Tuple[str, float]] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
    ):
        random_state = np.random.RandomState(self.random_state)

        n_samples, self.n_features_in_ = X.shape
        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        if self.class_weight is not None:
            y_original = np.copy(y)

        y_encoded = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_encoded

        if self.class_weight is not None:
            # expanded_class_weight = compute_sample_weight(
            #     self.class_weight, y_original
            # )
            pass

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
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
