from cython cimport final
from libc.math cimport isnan, log2
from libc.stdlib cimport qsort
from libc.string cimport memcpy

import numpy as np
from scipy.sparse import issparse

cdef float32_t INFINITY_32t = np.inf

@final
cdef class DensePartitioner:
    def __init__(
        self,
        const float32_t[:, :] X,
        intp_t[::1] samples,
        float32_t[::1] feature_values,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values