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

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)
                
    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        float64_t : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

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

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end
        _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
        )
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        # after reset, pos = start
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef intp_t c
        cdef float64_t w = 1.0
        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                self.sum_left[<intp_t> self.y[i, 0]] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]
                # After reverse_reset, left = sum_total.
                # Substract right part will get left.
                self.sum_left[<intp_t> self.y[i, 0]] -= w

                self.weighted_n_left -= w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for c in range(self.n_classes):
            self.sum_right[c] = self.sum_total[c] - self.sum_left[c]

        self.pos = new_pos
        return 0

    cdef void node_value(
        self,
        float64_t* dest
    ) noexcept nogil:
        cdef intp_t c

        for c in range(self.n_classes):
            dest[c] = self.sum_total[c] / self.weighted_n_node_samples

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

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address to save the impurity of the left node to
        impurity_right : float64_t pointer
            The memory address to save the impurity of the right node to
        """
        cdef float64_t gini_left = 0.0
        cdef float64_t gini_right = 0.0
        cdef float64_t sq_count_left
        cdef float64_t sq_count_right
        cdef float64_t count_c
        cdef intp_t c


        sq_count_left = 0.0
        sq_count_right = 0.0

        for c in range(self.n_classes):
            count_c = self.sum_left[c]
            sq_count_left += count_c * count_c

            count_c = self.sum_right[c]
            sq_count_right += count_c * count_c

        gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                            self.weighted_n_left)

        gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                self.weighted_n_right)

        impurity_left[0] = gini_left
        impurity_right[0] = gini_right

cdef class Entropy(Criterion):
    pass


