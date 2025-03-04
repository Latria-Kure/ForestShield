from typing import List, Tuple
from numbers import Integral, Real
from forest_shield.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import numpy as np


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


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = np.random.RandomState(random_state)
    sample_indices = random_instance.randint(
        0, n_samples, n_samples_bootstrap, dtype=np.int32
    )

    return sample_indices


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
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
            sample_weight=curr_sample_weight,
            check_input=False,
        )
    else:
        tree._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
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
        monotonic_cst: List[Tuple[str, float]] = None,
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
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        self._n_samples, self.n_outputs_ = y.shape
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
                    random_state=self.random_state,
                    ccp_alpha=self.ccp_alpha,
                    monotonic_cst=self.monotonic_cst,
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
