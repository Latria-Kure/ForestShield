cimport numpy as cnp
from ._tree cimport Node

from ..utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, int32_t, uint32_t

cdef enum:
    RAND_R_MAX = 2147483647

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (uint8_t*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil

cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil


cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil