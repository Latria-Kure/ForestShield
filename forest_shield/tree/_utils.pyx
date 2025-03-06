from libc.stdlib cimport free
from libc.stdlib cimport realloc

cdef inline uint32_t DEFAULT_SEED = 1

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError(f"could not allocate ({nelems} * {sizeof(p[0][0])}) bytes")

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError(f"could not allocate {nbytes} bytes")

    p[0] = tmp
    return 0

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0):
        seed[0] = DEFAULT_SEED

    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    # Use the modulo to make sure that we don't return a values greater than the
    # maximum representable value for signed 32bit integers (i.e. 2^31 - 1).
    # Note that the parenthesis are needed to avoid overflow: here
    # RAND_R_MAX is cast to uint32_t before 1 is added.
    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)

cdef inline intp_t rand_int(intp_t low, intp_t high,
                            uint32_t* random_state) noexcept nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


cdef inline float64_t rand_uniform(float64_t low, float64_t high,
                                   uint32_t* random_state) noexcept nogil:
    """Generate a random float64_t in [low; high)."""
    return ((high - low) * <float64_t> our_rand_r(random_state) /
            <float64_t> RAND_R_MAX) + low