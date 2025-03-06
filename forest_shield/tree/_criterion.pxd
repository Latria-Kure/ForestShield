from ..utils._typedefs cimport float64_t, int8_t, intp_t

cdef class Criterion:
    """Base criterion class for decision tree learning.
    
    This class provides a simple method to add two numbers.
    """
    cdef const float64_t[:,::1] y         # Values of y (one column vector)
    cdef const float64_t[:] sample_weight  # Sample weights

    cdef const intp_t[:] sample_indices    # Sample indices in X, y
    cdef intp_t start                      # samples[start:pos] are the samples in the left node
    cdef intp_t pos                        # samples[pos:end] are the samples in the right node
    cdef intp_t end

    cdef intp_t n_samples                  # Number of samples
    cdef intp_t n_node_samples             # Number of samples in the node (end-start)
    cdef float64_t weighted_n_samples         # Weighted number of samples (in total)
    cdef float64_t weighted_n_node_samples    # Weighted number of samples in the node
    cdef float64_t weighted_n_left            # Weighted number of samples in the left node
    cdef float64_t weighted_n_right           # Weighted number of samples in the right node

    cdef intp_t n_classes

    cdef float64_t[::1] sum_total    # The sum of the weighted count of each label.
    cdef float64_t[::1] sum_left     # Same as above, but for the left side of the split
    cdef float64_t[::1] sum_right    # Same as above, but for the right side of the split
    cdef float64_t[::1] sum_missing  # Same as above, but for missing values in X

    cdef int init(
        self,
        const float64_t[:,::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil
    cdef int reset(self) except -1 nogil
    cdef float64_t node_impurity(self) noexcept nogil

