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

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil:
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            intp_t i
            float32_t[::1] feature_values = self.feature_values
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples

        for i in range(self.start, self.end):
            feature_values[i] = X[samples[i], current_feature]

        sort(&feature_values[self.start], &samples[self.start], self.end - self.start)

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iterating over feature values.

        The missing values are not included when iterating through the feature values.
        """
        cdef:
            float32_t[::1] feature_values = self.feature_values
            intp_t end = self.end

        # stop when feature[index] < feature[index + 1]
        while (
            p[0] + 1 < end and
            feature_values[p[0] + 1] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] += 1

        p_prev[0] = p[0]

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
    ) noexcept nogil:
        cdef:
            intp_t start = self.start
            intp_t p = start
            intp_t end = self.end - 1
            intp_t[::1] samples = self.samples
            const float32_t[:, :] X = self.X
            float32_t current_value

        while p < end:
            if X[samples[p], best_feature] <= best_threshold:
                p += 1
            else:
                samples[p], samples[end] = samples[end], samples[p]
                end -= 1

# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    if n == 0:
        return
    cdef intp_t maxd = 2 * <intp_t>log2(n)
    introsort(feature_values, samples, n, maxd)

cdef inline void swap(float32_t* feature_values, intp_t* samples,
                      intp_t i, intp_t j) noexcept nogil:
    # Helper for sort
    feature_values[i], feature_values[j] = feature_values[j], feature_values[i]
    samples[i], samples[j] = samples[j], samples[i]

cdef inline float32_t median3(float32_t* feature_values, intp_t n) noexcept nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef float32_t a = feature_values[0], b = feature_values[n // 2], c = feature_values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(float32_t* feature_values, intp_t *samples,
                    intp_t n, intp_t maxd) noexcept nogil:
    cdef float32_t pivot
    cdef intp_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        pivot = median3(feature_values, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, l)
                i += 1
                l += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        introsort(feature_values, samples, l, maxd)
        feature_values += r
        samples += r
        n -= r

cdef inline void sift_down(float32_t* feature_values, intp_t* samples,
                           intp_t start, intp_t end) noexcept nogil:
    # Restore heap order in feature_values[start:end] by moving the max element to start.
    cdef intp_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and feature_values[maxind] < feature_values[child]:
            maxind = child
        if child + 1 < end and feature_values[maxind] < feature_values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(feature_values, samples, root, maxind)
            root = maxind

cdef void heapsort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    cdef intp_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(feature_values, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(feature_values, samples, 0, end)
        sift_down(feature_values, samples, 0, end)
        end = end - 1
