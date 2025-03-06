cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.string cimport memcpy
from libc.string cimport memset


cdef inline void _move_sums_classification(
    Criterion criterion,
    float64_t[::1] sum_1,
    float64_t[::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
) noexcept nogil:
    """
    Distribute sum_total and sum_missing into sum_1 and sum_2.
    """
    cdef intp_t k, c, n_bytes
    n_bytes = criterion.n_classes * sizeof(float64_t)
    memset(&sum_1[0], 0, n_bytes)
    memcpy(&sum_2[0], &criterion.sum_total[0], n_bytes)

    weighted_n_1[0] = 0.0
    weighted_n_2[0] = criterion.weighted_n_node_samples

cdef class Criterion:
    """Base criterion class for decision tree learning.
    
    This class provides a simple method to add two numbers.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    def __cinit__(self,intp_t n_classes):
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        
        self.n_classes = n_classes

        self.sum_total = np.zeros(n_classes, dtype=np.float64)
        self.sum_left = np.zeros(n_classes, dtype=np.float64)
        self.sum_right = np.zeros(n_classes, dtype=np.float64)
    
    def __reduce__(self):
        return (type(self), (self.n_classes,), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        memset(&self.sum_total[0], 0, self.n_classes * sizeof(float64_t))

        cdef intp_t i
        cdef intp_t p
        cdef float64_t w = 1.0
        
        for p in range(start, end):
            i = sample_indices[p]
            if sample_weight is not None:
                w = sample_weight[i]
            self.sum_total[<intp_t> y[i, 0]] += w

            self.weighted_n_node_samples += w

        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start
        _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
        )
        return 0

cdef class Gini(Criterion):
    cdef float64_t node_impurity(self) noexcept nogil:
        cdef float64_t gini = 0.0
        cdef float64_t sq_count
        cdef float64_t count
        cdef intp_t c

        for c in range(self.n_classes):
            count = self.sum_total[c]
            sq_count = count * count

        gini += 1.0 - sq_count / (self.weighted_n_node_samples * self.weighted_n_node_samples)

        return gini

cdef class Entropy(Criterion):
    pass


