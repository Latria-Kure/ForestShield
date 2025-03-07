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

    # Methods
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    cdef void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil
    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil
    cdef void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature
    ) noexcept nogil

cdef class SparsePartitioner:
    pass



