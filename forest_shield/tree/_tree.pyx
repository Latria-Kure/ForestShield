from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef class TreeBuilder:
    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:,::1] y,
        const float64_t[:] sample_weight=None
    ):
        pass

cdef struct StackRecord:
    intp_t start
    intp_t end
    intp_t depth
    intp_t parent
    bint is_left
    float64_t impurity
    intp_t n_constant_features
    float64_t lower_bound
    float64_t upper_bound

cdef class DepthFirstTreeBuilder(TreeBuilder):
    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

cdef class BestFirstTreeBuilder(TreeBuilder):
    pass

cdef class Tree:
    pass
