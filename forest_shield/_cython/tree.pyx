"""
Optimized Cython implementation of decision tree.
"""

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log2, fmax, sqrt, fmin

# Define types
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

# Define constants
cdef double INFINITY = float('inf')
cdef double EPSILON = np.finfo('double').eps


cdef struct Node:
    int feature
    double threshold
    int left_child
    int right_child
    double* value
    int is_leaf


cdef class CythonTree:
    """Cython implementation of a decision tree."""
    
    cdef:
        int max_depth
        int min_samples_split
        int min_samples_leaf
        int max_features
        int criterion  # 0 for gini, 1 for entropy
        int random_state
        int n_features
        int n_classes
        int node_count
        int capacity
        Node* nodes
        double* feature_importances
    
    def __cinit__(self, int max_depth=-1, int min_samples_split=2, int min_samples_leaf=1,
                 int max_features=-1, int criterion=0, int random_state=0):
        """Initialize the tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.node_count = 0
        self.capacity = 10  # Initial capacity
        self.nodes = <Node*>malloc(self.capacity * sizeof(Node))
        self.feature_importances = NULL
    
    def __dealloc__(self):
        """Free memory."""
        if self.nodes != NULL:
            # Free value arrays for each node
            for i in range(self.node_count):
                if self.nodes[i].value != NULL:
                    free(self.nodes[i].value)
            free(self.nodes)
        
        if self.feature_importances != NULL:
            free(self.feature_importances)
    
    cdef void _resize(self, int capacity):
        """Resize the nodes array."""
        cdef Node* new_nodes = <Node*>malloc(capacity * sizeof(Node))
        
        # Copy existing nodes
        for i in range(self.node_count):
            new_nodes[i] = self.nodes[i]
        
        # Free old array and update
        free(self.nodes)
        self.nodes = new_nodes
        self.capacity = capacity
    
    cdef int _add_node(self, int feature, double threshold, int left_child, int right_child, 
                      double* value, int is_leaf):
        """Add a new node to the tree."""
        cdef int node_id = self.node_count
        
        # Resize if needed
        if self.node_count >= self.capacity:
            self._resize(self.capacity * 2)
        
        # Initialize the new node
        self.nodes[node_id].feature = feature
        self.nodes[node_id].threshold = threshold
        self.nodes[node_id].left_child = left_child
        self.nodes[node_id].right_child = right_child
        self.nodes[node_id].value = value
        self.nodes[node_id].is_leaf = is_leaf
        
        # Increment node count
        self.node_count += 1
        
        return node_id
    
    cdef double _calculate_impurity(self, ITYPE_t[:] y, int start, int end):
        """Calculate impurity of a node."""
        cdef:
            int i, j, count
            int n_samples = end - start
            double* class_counts
            double impurity = 0.0
            double p
        
        if n_samples == 0:
            return 0.0
        
        # Allocate memory for class counts
        class_counts = <double*>malloc(self.n_classes * sizeof(double))
        for i in range(self.n_classes):
            class_counts[i] = 0.0
        
        # Count classes
        for i in range(start, end):
            class_counts[y[i]] += 1.0
        
        # Calculate impurity
        if self.criterion == 0:  # Gini
            impurity = 1.0
            for i in range(self.n_classes):
                p = class_counts[i] / n_samples
                impurity -= p * p
        else:  # Entropy
            for i in range(self.n_classes):
                if class_counts[i] > 0:
                    p = class_counts[i] / n_samples
                    impurity -= p * log2(p)
        
        # Free memory
        free(class_counts)
        
        return impurity
    
    cdef double _calculate_information_gain(self, ITYPE_t[:] y, int start, int end, 
                                          int left_start, int left_end, int right_start, int right_end):
        """Calculate information gain from a split."""
        cdef:
            double parent_impurity
            double left_impurity
            double right_impurity
            int n_samples = end - start
            int n_left = left_end - left_start
            int n_right = right_end - right_start
            double p_left = n_left / <double>n_samples
        
        parent_impurity = self._calculate_impurity(y, start, end)
        left_impurity = self._calculate_impurity(y, left_start, left_end)
        right_impurity = self._calculate_impurity(y, right_start, right_end)
        
        return parent_impurity - p_left * left_impurity - (1.0 - p_left) * right_impurity
    
    cdef void _find_best_split(self, DTYPE_t[:, :] X, ITYPE_t[:] y, int start, int end, 
                             int* features, int n_features_to_consider, 
                             int* best_feature, double* best_threshold, double* best_info_gain):
        """Find the best split for a node."""
        cdef:
            int i, j, feature, n_samples, n_left, n_right
            double threshold, info_gain
            int left_start, left_end, right_start, right_end
            int* indices = <int*>malloc((end - start) * sizeof(int))
            double* values = <double*>malloc((end - start) * sizeof(double))
        
        # Initialize
        best_info_gain[0] = -INFINITY
        best_feature[0] = -1
        best_threshold[0] = 0.0
        n_samples = end - start
        
        # Try each feature
        for i in range(n_features_to_consider):
            feature = features[i]
            
            # Get values for this feature
            for j in range(start, end):
                indices[j - start] = j
                values[j - start] = X[j, feature]
            
            # Sort values
            self._sort_indices_and_values(indices, values, 0, n_samples - 1)
            
            # Try each threshold
            for j in range(n_samples - 1):
                # Skip if values are identical
                if values[j] == values[j + 1]:
                    continue
                
                # Calculate threshold
                threshold = (values[j] + values[j + 1]) / 2.0
                
                # Count samples in left and right
                left_start = start
                left_end = start + j + 1
                right_start = left_end
                right_end = end
                
                n_left = left_end - left_start
                n_right = right_end - right_start
                
                # Skip if split doesn't meet min_samples_leaf
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                info_gain = self._calculate_information_gain(
                    y, start, end, left_start, left_end, right_start, right_end
                )
                
                # Update best split
                if info_gain > best_info_gain[0]:
                    best_info_gain[0] = info_gain
                    best_feature[0] = feature
                    best_threshold[0] = threshold
        
        # Free memory
        free(indices)
        free(values)
    
    cdef void _sort_indices_and_values(self, int* indices, double* values, int start, int end):
        """Sort indices and values using quicksort."""
        cdef:
            int i, j, temp_idx
            double pivot, temp_val
        
        if start >= end:
            return
        
        # Choose pivot
        pivot = values[end]
        i = start - 1
        
        # Partition
        for j in range(start, end):
            if values[j] <= pivot:
                i += 1
                # Swap indices
                temp_idx = indices[i]
                indices[i] = indices[j]
                indices[j] = temp_idx
                # Swap values
                temp_val = values[i]
                values[i] = values[j]
                values[j] = temp_val
        
        # Swap pivot
        i += 1
        temp_idx = indices[i]
        indices[i] = indices[end]
        indices[end] = temp_idx
        temp_val = values[i]
        values[i] = values[end]
        values[end] = temp_val
        
        # Recursively sort
        self._sort_indices_and_values(indices, values, start, i - 1)
        self._sort_indices_and_values(indices, values, i + 1, end)
    
    cdef int _build_tree(self, DTYPE_t[:, :] X, ITYPE_t[:] y, int start, int end, int depth):
        """Recursively build the decision tree."""
        cdef:
            int n_samples = end - start
            int n_features_total = X.shape[1]
            int n_features_to_consider
            int* features
            int best_feature = -1
            double best_threshold = 0.0
            double best_info_gain = -INFINITY
            int left_child, right_child
            double* value
            int i, j, count
            int is_pure = 1
            int first_class
            int* partition_indices
            int left_start, left_end, right_start, right_end
        
        # Check if we should stop splitting
        if (self.max_depth >= 0 and depth >= self.max_depth) or n_samples < self.min_samples_split:
            # Create a leaf node
            value = <double*>malloc(self.n_classes * sizeof(double))
            for i in range(self.n_classes):
                value[i] = 0.0
            
            # Count classes
            for i in range(start, end):
                value[y[i]] += 1.0
            
            # Convert to probabilities
            for i in range(self.n_classes):
                value[i] /= n_samples
            
            return self._add_node(-1, 0.0, -1, -1, value, 1)
        
        # Check if node is pure
        if n_samples > 0:
            first_class = y[start]
            for i in range(start + 1, end):
                if y[i] != first_class:
                    is_pure = 0
                    break
        
        if is_pure:
            # Create a leaf node
            value = <double*>malloc(self.n_classes * sizeof(double))
            for i in range(self.n_classes):
                value[i] = 0.0
            
            # Set probability to 1.0 for the class
            value[y[start]] = 1.0
            
            return self._add_node(-1, 0.0, -1, -1, value, 1)
        
        # Determine features to consider
        if self.max_features <= 0:
            n_features_to_consider = n_features_total
        elif self.max_features == 1:  # sqrt
            n_features_to_consider = <int>fmax(1, sqrt(n_features_total))
        elif self.max_features == 2:  # log2
            n_features_to_consider = <int>fmax(1, log2(n_features_total))
        else:
            n_features_to_consider = <int>fmax(1, fmin(self.max_features, n_features_total))
        
        # Randomly select features
        features = <int*>malloc(n_features_to_consider * sizeof(int))
        self._random_features(n_features_total, n_features_to_consider, features)
        
        # Find the best split
        self._find_best_split(X, y, start, end, features, n_features_to_consider, 
                            &best_feature, &best_threshold, &best_info_gain)
        
        # Free memory
        free(features)
        
        # If no valid split found, create a leaf node
        if best_feature < 0:
            # Create a leaf node
            value = <double*>malloc(self.n_classes * sizeof(double))
            for i in range(self.n_classes):
                value[i] = 0.0
            
            # Count classes
            for i in range(start, end):
                value[y[i]] += 1.0
            
            # Convert to probabilities
            for i in range(self.n_classes):
                value[i] /= n_samples
            
            return self._add_node(-1, 0.0, -1, -1, value, 1)
        
        # Partition the data
        partition_indices = <int*>malloc(n_samples * sizeof(int))
        for i in range(start, end):
            partition_indices[i - start] = i
        
        # Partition based on the best split
        left_start = start
        right_end = end
        j = 0
        for i in range(start, end):
            if X[i, best_feature] <= best_threshold:
                partition_indices[j] = i
                j += 1
        
        left_end = start + j
        right_start = left_end
        
        # Create a decision node
        node_id = self._add_node(best_feature, best_threshold, -1, -1, NULL, 0)
        
        # Build left and right subtrees
        left_child = self._build_tree(X, y, left_start, left_end, depth + 1)
        right_child = self._build_tree(X, y, right_start, right_end, depth + 1)
        
        # Update node with children
        self.nodes[node_id].left_child = left_child
        self.nodes[node_id].right_child = right_child
        
        # Free memory
        free(partition_indices)
        
        return node_id
    
    cdef void _random_features(self, int n_features, int n_features_to_consider, int* features):
        """Randomly select features without replacement."""
        cdef:
            int i, j, temp
            int* all_features = <int*>malloc(n_features * sizeof(int))
        
        # Initialize all features
        for i in range(n_features):
            all_features[i] = i
        
        # Fisher-Yates shuffle
        for i in range(n_features - 1, n_features - n_features_to_consider - 1, -1):
            j = self._rand_int(0, i + 1)
            temp = all_features[i]
            all_features[i] = all_features[j]
            all_features[j] = temp
        
        # Copy selected features
        for i in range(n_features_to_consider):
            features[i] = all_features[n_features - i - 1]
        
        # Free memory
        free(all_features)
    
    cdef int _rand_int(self, int low, int high):
        """Generate a random integer in [low, high)."""
        return low + (self.random_state % (high - low))
    
    cdef void _update_random_state(self):
        """Update the random state."""
        self.random_state = (1103515245 * self.random_state + 12345) & 0x7fffffff
    
    def fit(self, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[ITYPE_t, ndim=1] y):
        """Build a decision tree classifier from the training set (X, y)."""
        cdef:
            int n_samples = X.shape[0]
            int n_features = X.shape[1]
            int i
        
        # Initialize
        self.n_features = n_features
        self.n_classes = np.max(y) + 1
        
        # Build the tree
        self._build_tree(X, y, 0, n_samples, 0)
        
        # Compute feature importances
        self.feature_importances = <double*>malloc(n_features * sizeof(double))
        for i in range(n_features):
            self.feature_importances[i] = 0.0
        
        # TODO: Implement feature importance calculation
        
        return self
    
    def predict_proba(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Predict class probabilities of the input samples X."""
        cdef:
            int n_samples = X.shape[0]
            int i
            np.ndarray[DTYPE_t, ndim=2] proba = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        
        # Predict for each sample
        for i in range(n_samples):
            self._predict_proba_sample(X[i], 0, &proba[i, 0])
        
        return proba
    
    cdef void _predict_proba_sample(self, DTYPE_t[:] x, int node_id, double* out):
        """Predict class probabilities for a single sample."""
        cdef:
            int feature
            double threshold
            int i
        
        if self.nodes[node_id].is_leaf:
            # Copy probabilities
            for i in range(self.n_classes):
                out[i] = self.nodes[node_id].value[i]
            return
        
        feature = self.nodes[node_id].feature
        threshold = self.nodes[node_id].threshold
        
        if x[feature] <= threshold:
            self._predict_proba_sample(x, self.nodes[node_id].left_child, out)
        else:
            self._predict_proba_sample(x, self.nodes[node_id].right_child, out)
    
    def predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Predict class for X."""
        cdef:
            int n_samples = X.shape[0]
            np.ndarray[ITYPE_t, ndim=1] predictions = np.zeros(n_samples, dtype=np.int32)
            np.ndarray[DTYPE_t, ndim=2] proba = self.predict_proba(X)
            int i
        
        # Get class with highest probability
        for i in range(n_samples):
            predictions[i] = np.argmax(proba[i])
        
        return predictions
    
    def get_feature_importances(self):
        """Get feature importances."""
        cdef:
            np.ndarray[DTYPE_t, ndim=1] importances = np.zeros(self.n_features, dtype=np.float64)
            int i
            double normalizer = 0.0
        
        # Copy feature importances
        for i in range(self.n_features):
            importances[i] = self.feature_importances[i]
            normalizer += importances[i]
        
        # Normalize
        if normalizer > 0:
            for i in range(self.n_features):
                importances[i] /= normalizer
        
        return importances 