cimport cython
from libc.math cimport sqrt


ctypedef fused my_float:
    float
    double


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def sum_inv_pairdists(my_float [:,::1] pos):
    """Calculate the sum of inverse pair distances for a collection of particles.
    Can be used to calculate potentials / potential energies.
    """
    cdef int N = pos.shape[0]
    cdef int dims = pos.shape[1]
    cdef double potential = 0

    cdef int i, j, k
    cdef my_float dist2, temp

    for i in range(N):
        for j in range(i):
            dist2 = 0
            for k in range(dims):
                temp = pos[i,k] - pos[j,k]
                dist2 += temp * temp
            potential -= 1/sqrt(dist2 + 1e-14)

    return potential
