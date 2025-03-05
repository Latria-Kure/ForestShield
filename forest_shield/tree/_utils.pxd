cimport numpy as cnp
from ._tree cimport Node

from ..utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, int32_t, uint32_t

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