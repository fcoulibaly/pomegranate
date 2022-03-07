# TrueBetaDistribution.pxd
# Contact: Pierre Louvart <pierre.louvart@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TrueBetaDistribution(Distribution):
	cdef double alpha, beta

