cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import numpy as np
cimport numpy as cnp

from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object
    intp_t left_child                    # id of the left child of the node
    intp_t right_child                   # id of the right child of the node
    intp_t feature                       # Feature used for splitting the node
    float64_t threshold                  # Threshold value at the node
    float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    intp_t n_node_samples                # Number of samples at the node
    float64_t weighted_n_node_samples    # Weighted number of samples at the node

cdef struct ParentInfo:
    # Structure to store information about the parent of a node
    # This is passed to the splitter, to provide information about the previous split

    float64_t lower_bound           # the lower bound of the parent's impurity
    float64_t upper_bound           # the upper bound of the parent's impurity
    float64_t impurity              # the impurity of the parent
    intp_t n_constant_features      # the number of constant features found in parent

cdef class Tree:
    # Input/Output layout
    cdef public intp_t n_features        # Number of features in X
    cdef intp_t n_classes               # Number of classes in y[:, k]

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public intp_t max_depth         # Max depth of the tree
    cdef public intp_t node_count        # Counter for node IDs
    cdef public intp_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef float64_t* value                # (capacity, n_classes) array of values
    cdef intp_t value_stride             # = n_classes

    # Methods
    cdef int _resize(self, intp_t capacity) except -1 nogil
    cdef int _resize_c(self, intp_t capacity=*) except -1 nogil
    cdef cnp.ndarray _get_value_ndarray(self)
    cdef cnp.ndarray _get_node_ndarray(self)
    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples) except -1 nogil
    cpdef cnp.ndarray predict(self, object X)
    cpdef cnp.ndarray apply(self, object X)

cdef class TreeBuilder:
    cdef Splitter splitter
    cdef intp_t min_samples_split       # Minimum number of samples in an internal node
    cdef intp_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef float64_t min_weight_leaf         # Minimum weight in a leaf
    cdef intp_t max_depth               # Maximal tree depth
    cdef float64_t min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:,::1] y,
        const float64_t[:] sample_weight=*
    )
    cdef _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    )


