# TrueBetaDistribution.pxd
# Contact: Pierre Louvart <pierre.louvart@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TrueBetaDistribution(Distribution):
	cdef double alpha, beta, one_minus_alpha, one_minus_beta, lgamma_alpha_beta
	cdef object min_std

