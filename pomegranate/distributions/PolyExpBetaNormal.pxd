# MultivariateGaussianDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from ..base cimport Model

from .distributions cimport Distribution
from .distributions cimport MultivariateDistribution

cdef enum:
    nb_bins = 32

cdef struct Wcparams:
    double sw
    double sy
    double sy2
    double sws[nb_bins]
    double sys[nb_bins]
    double xms[nb_bins]

cdef class PolyExpBetaNormal(MultivariateDistribution):
	cdef public numpy.ndarray acoeffs, bcoeffs, ccoeffs, sigmas
        cdef public alpha, beta
        cdef double _alpha
        cdef double _beta
        cdef double* _acoeffs
        cdef double* _bcoeffs
        cdef double* _ccoeffs
        cdef double* _sigmas
