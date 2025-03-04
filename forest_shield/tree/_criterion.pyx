
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef class Criterion:
    """Base criterion class for decision tree learning.
    
    This class provides a simple method to add two numbers.
    """
    
    def __init__(self):
        """Initialize the criterion."""
        pass
    
    cpdef double add(self, double a, double b):
        """Add two numbers and return the result.
        
        Parameters
        ----------
        a : double
            First number.
        b : double
            Second number.
            
        Returns
        -------
        double
            The sum of a and b.
        """
        return a + b
    
cdef class Entropy(Criterion):
    def __cinit__(self, intp_t n_outputs,
                  cnp.ndarray[intp_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

