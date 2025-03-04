
import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.string cimport memset

cdef class Criterion:
    """Base criterion class for decision tree learning.
    
    This class provides a simple method to add two numbers.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
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
        const float64_t[::1] y,
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
            self.sum_total[<intp_t> y[i]] += w

            self.weighted_n_samples += w

        # self.reset()
        return 0

cdef class Gini(Criterion):
    pass

cdef class Entropy(Criterion):
    pass


