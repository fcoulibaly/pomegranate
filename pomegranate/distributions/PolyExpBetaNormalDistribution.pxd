# PolyExpBetaNormalDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from ..base cimport Model

from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution

cdef class PolyExpBetaNormalDistribution(MultivariateDistribution):
    cdef public numpy.ndarray coeffs, sigmas, lower_bounds, upper_bounds
    cdef public numpy.ndarray acoeffs, bcoeffs, ccoeffs
    cdef public numpy.ndarray infered_coeffs, infered_sigmas

    cdef int has_apriori

    cdef double _alpha
    cdef double _beta
    cdef double lgamma_constants
    cdef double* log_sigma_sqrt_2_pi
    cdef double* two_sigma_squared
    cdef double* _coeffs
    cdef double* _acoeffs
    cdef double* _bcoeffs
    cdef double* _ccoeffs
    cdef double* _sigmas
    cdef double* _bounds

    cdef double* _infered_coeffs
    cdef double _infered_alpha
    cdef double _infered_beta
    cdef double* _infered_sigmas
    cdef double* _lower_bounds
    cdef double* _upper_bounds

    cdef double _log_probability_missing(self, double* X) nogil

