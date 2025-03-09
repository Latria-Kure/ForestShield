from typing import List, Tuple
from numbers import Integral, Real
from forest_shield.tree import DecisionTreeClassifier
from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
import pandas as pd
import time
import threading

from ..tree._tree import DOUBLE, DTYPE


def _get_n_samples_bootstrap(n_samples, max_samples):
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return max(round(n_samples * max_samples), 1)


def _generate_sample_indices(
    random_state: np.random.RandomState, n_samples, n_samples_bootstrap
):
    """
    Private function used to _parallel_build_trees function."""

    sample_indices = random_state.randint(
        0, n_samples, n_samples_bootstrap, dtype=np.int32
    )

    return sample_indices


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X)
    with lock:
        out += prediction


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    classes_,
    n_classes_,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            # curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
            pass
        elif class_weight == "balanced_subsample":
            # curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)
            pass
        tree._fit(
            X,
            y,
            classes_,
            n_classes_,
            sample_weight=curr_sample_weight,
        )

    else:
        tree._fit(
            X,
            y,
            classes_,
            n_classes_,
            sample_weight=sample_weight,
        )

    return tree


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str = "sqrt",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        class_weight: dict = None,
        ccp_alpha: float = 0.0,
        max_samples: int = None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = np.random.RandomState(random_state)
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_samples = max_samples

    def _validate_y_class_weight(self, y):
        y = np.copy(y)
        expanded_class_weight = None
        if self.class_weight is not None:
            y_original = np.copy(y)
        y_store_unique_indices = np.zeros(y.shape, dtype=int)

        self.classes_, y_store_unique_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        y = y_store_unique_indices

        # TODO: Implement class weight
        return y, expanded_class_weight

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        X = X.values.astype(DTYPE)
        y = y.values

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        self._n_samples = y.shape[0]

        y, expanded_class_weight = self._validate_y_class_weight(y)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            # TODO: Implement class weight
            pass

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        self.estimators_ = []
        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators=%d must be larger than 0" % self.n_estimators
            )
        else:
            trees = [
                DecisionTreeClassifier(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                    random_state=self.random_state.randint(np.iinfo(np.int32).max),
                    ccp_alpha=self.ccp_alpha,
                )
                for i in range(self.n_estimators)
            ]

            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    self.classes_,
                    self.n_classes_,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            self.estimators_.extend(trees)

        if self.oob_score and (
            self.n_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            pass

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in the forest,
        weighted by the sample weight when possible.
        """
        X = X.values.astype(DTYPE)
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(
                e.predict_proba,
                X,
                all_proba,
                lock,
            )
            for e in self.estimators_
        )

        all_proba /= len(self.estimators_)
        return all_proba
