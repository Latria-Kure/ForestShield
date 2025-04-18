from cpython cimport Py_INCREF, PyObject, PyTypeObject

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport INTPTR_MAX
from libcpp.stack cimport stack
from libc.stdio cimport printf
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._utils cimport safe_realloc
# Types and constants
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef float64_t INFINITY = np.inf
cdef float64_t EPSILON = np.finfo('double').eps


TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef Node dummy
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef inline void _init_parent_record(ParentInfo* record) noexcept nogil:
    record.n_constant_features = 0
    record.impurity = INFINITY
    record.lower_bound = -INFINITY
    record.upper_bound = INFINITY

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)
cdef class TreeBuilder:
    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:,::1] y,
        const float64_t[:] sample_weight=None
    ):
        pass
    cdef _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    ):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)


        if (
            sample_weight is not None and
            (
                sample_weight.base.dtype != DOUBLE or
                not sample_weight.base.flags.contiguous
            )
        ):
            sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")

        return X, y, sample_weight

cdef struct StackRecord:
    intp_t start
    intp_t end
    intp_t depth
    intp_t parent
    bint is_left
    float64_t impurity
    intp_t n_constant_features

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

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
    ):
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        # Initial capacity
        cdef intp_t init_capacity

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        cdef Splitter splitter = self.splitter
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        splitter.init(X, y, sample_weight)
        cdef intp_t start
        cdef intp_t end
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef SplitRecord split
        cdef intp_t node_id

        cdef float64_t middle_value
        cdef bint is_leaf
        cdef bint first = 1
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord] builder_stack
        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)
        with nogil:
            builder_stack.push({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_features": 0
            })

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                parent_record.impurity = stack_record.impurity
                parent_record.n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    parent_record.impurity = splitter.node_impurity()
                    first = 0

                is_leaf = is_leaf or parent_record.impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(
                        &parent_record,
                        &split,
                    )

                    is_leaf = (is_leaf or split.pos >= end or
                        (split.improvement + EPSILON <
                        min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, parent_record.impurity,
                                         n_node_samples, weighted_n_node_samples)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # right
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": parent_record.n_constant_features
                    })

                    # left
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": parent_record.n_constant_features
                    })

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

cdef class BestFirstTreeBuilder(TreeBuilder):
    pass

cdef class Tree:
    def __cinit__(self, intp_t n_features, intp_t n_classes):
        self.n_features = n_features
        self.n_classes = n_classes

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL
        self.value_stride = self.n_classes

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.value)
        free(self.nodes)

    cdef cnp.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef cnp.npy_intp shape[2]
        shape[0] = <cnp.npy_intp> self.node_count
        shape[1] = <cnp.npy_intp> self.n_classes 
        cdef cnp.ndarray arr
        arr = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef cnp.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> cnp.ndarray,
                                   <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   cnp.NPY_ARRAY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features, self.n_classes), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        memcpy(self.nodes, cnp.PyArray_DATA(node_ndarray),
               self.capacity * sizeof(Node))
        memcpy(self.value, cnp.PyArray_DATA(value_ndarray),
               self.capacity * self.value_stride * sizeof(float64_t))

    cdef int _resize(self, intp_t capacity) except -1 nogil:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, intp_t capacity=INTPTR_MAX) except -1 nogil:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        if capacity > self.capacity:
            # value memory is initialised to 0 to enable classifier argmax
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(float64_t))
            # node memory is initialised to 0 to ensure deterministic pickle (padding in Node struct)
            memset(<void*>(self.nodes + self.capacity), 0, (capacity - self.capacity) * sizeof(Node))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0
    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples) except -1 nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef intp_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return INTPTR_MAX

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef cnp.ndarray predict(self, object X):
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        out = out.reshape(X.shape[0], self.n_classes)
        return out

    cpdef cnp.ndarray apply(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        
        cdef const float32_t[:, :] X_ndarray = X
        cdef intp_t n_samples = X.shape[0]
        cdef float32_t X_i_node_feature

        cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

        cdef Node* node = NULL
        cdef intp_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    X_i_node_feature = X_ndarray[i, node.feature]
                    if X_i_node_feature <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out[i] = <intp_t>(node - self.nodes)  # node offset

        return np.asarray(out)

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef float64_t normalizer = 0.

        cdef cnp.float64_t[:] importances = np.zeros(self.n_features)

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importances[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        for i in range(self.n_features):
            importances[i] /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                for i in range(self.n_features):
                    importances[i] /= normalizer

        return np.asarray(importances)