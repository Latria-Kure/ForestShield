from ..utils._typedefs cimport (
    float32_t, float64_t, int8_t, int32_t, intp_t, uint8_t, uint32_t
)
from ._splitter cimport SplitRecord

cdef float32_t FEATURE_THRESHOLD = 1e-7

cdef class DensePartitioner:
    cdef const float32_t[:, :] X
    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end

cdef class SparsePartitioner:
    pass
