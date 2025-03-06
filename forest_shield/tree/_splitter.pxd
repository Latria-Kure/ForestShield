from ..utils._typedefs cimport (
    float32_t, float64_t, int8_t, int32_t, intp_t, uint8_t, uint32_t
)
from ._criterion cimport Criterion
from ._tree cimport ParentInfo

cdef struct SplitRecord:
    # Data to track sample split
    intp_t feature         # Which feature to split on.
    intp_t pos             # Split samples array at the given position,
    #                      # i.e. count of samples below threshold for feature.
    #                      # pos is >= end if the node is a leaf.
    float64_t threshold       # Threshold to split at.
    float64_t improvement     # Impurity improvement given parent node.
    float64_t impurity_left   # Impurity of the left split.
    float64_t impurity_right  # Impurity of the right split.
    float64_t lower_bound     # Lower bound on value of both children for monotonicity
    float64_t upper_bound     # Upper bound on value of both children for monotonicity

cdef class Splitter:
    cdef public Criterion criterion      # Impurity criterion
    cdef public intp_t max_features      # Number of features to test
    cdef public intp_t min_samples_leaf  # Min samples in a leaf
    cdef public float64_t min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef uint32_t rand_r_state

    cdef intp_t[::1] samples             # Sample indices in X, y
    cdef intp_t n_samples                # X.shape[0]
    cdef float64_t weighted_n_samples       # Weighted number of samples

    cdef intp_t[::1] features            # Feature indices in X
    cdef intp_t[::1] constant_features   # Constant features indices
    cdef intp_t n_features               # X.shape[1]
    cdef float32_t[::1] feature_values   # temp. array holding feature values

    cdef intp_t start                    # Start position for the current node
    cdef intp_t end                      # End position for the current node

    cdef const float64_t[:, ::1] y
    cdef const float64_t[:] sample_weight
    # Methods
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight
    ) except -1

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil

    cdef float64_t node_impurity(self) noexcept nogil