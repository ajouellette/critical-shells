cimport cython
from libc.math cimport sqrt


ctypedef fused my_float:
    float
    double


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def sum_inv_pairdists(my_float [:,::1] pos, epsilon=1e-2):
    """Calculate the sum of inverse pair distances for a collection of particles.
    Can be used to calculate potentials / potential energies.
    """
    cdef int N = pos.shape[0]
    cdef int dims = pos.shape[1]
    cdef double potential = 0

    cdef int i, j, k
    cdef my_float r, dist2, temp, h, u, W2, u2

    for i in range(N):
        for j in range(i):
            dist2 = 0
            for k in range(dims):
                temp = pos[i,k] - pos[j,k]
                dist2 += temp * temp

            r = sqrt(dist2)
            h = 2.8 * epsilon
            u = r / h

            if u < 0.5:
                #W2 = 16/3. * u**2 - 48/5. * u**4 + 32/5. * u**5 - 14/5.
                u2 = u * u
                W2 = -14/5. + u2 * (16/3. + u2 * (-48/5. + 32/5. * u))
            elif u < 1:
                #W2 = 1/(15 * u) + 32/3. * u**2 - 16 * u**3 + 48/5. * u**4 - 32/15. * u**5 - 16/5.
                W2 = 1/(15*u) - 16/5. + u*u * (32/3. + u * (-16 + u * (48/5. - 32/15. * u)))
            else:
                W2 = -1/u

            potential -= 1/(-h / W2)

    return potential
