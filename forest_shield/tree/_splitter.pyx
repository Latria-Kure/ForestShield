from ._utils cimport RAND_R_MAX, rand_int, rand_uniform
import numpy as np
cdef class Splitter:
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
    ):
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight
    ) except -1:
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        return 0

cdef class BestSplitter(Splitter):
    pass


cdef class BestSparseSplitter(Splitter):
    pass


cdef class RandomSplitter(Splitter):
    pass


cdef class RandomSparseSplitter(Splitter):
    pass

