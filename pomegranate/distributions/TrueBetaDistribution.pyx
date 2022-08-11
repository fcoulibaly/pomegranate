#!python
#cython: boundscheck=False
#cython: cdivision=True
# TrueBetaDistribution.pyx
# Contact: Pierre Louvart <pierre.louvart@gmail.com>

import numpy

from ..utils cimport _log
from ..utils cimport isnan
from ..utils cimport lgamma
from ..utils import check_random_state

cdef class TrueBetaDistribution(Distribution):
    """A beta distribution based on parameters alpha and beta defined on the unit domain."""

    property parameters:
        def __get__(self):
            return [self.alpha, self.beta]
        def __set__(self, parameters):
            self.alpha, self.beta = parameters

    def __init__(self, alpha, beta, frozen=False, min_std=0.0):
        self.alpha = alpha
        self.beta = beta
        self.name = "TrueBetaDistribution"
        self.frozen = frozen
        self.summaries = [0, 0, 0]
        self.one_minus_alpha = 1 - self.alpha
        self.one_minus_beta = 1 - self.beta
        self.lgamma_alpha_beta = lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta)
        self.min_std = min_std

    def __reduce__(self):
        """Serialize distribution for pickling."""
        return self.__class__, (self.alpha, self.beta, self.frozen)

    cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
        cdef int i
        for i in range(n):
            if isnan(X[i]) or X[i] <= 0 or X[i] >= 1:
                log_probability[i] = 0.
            else:
                log_probability[i] = _log(X[i]) * self.one_minus_alpha + _log(1 - X[i]) * self.one_minus_beta +\
                    self.lgamma_alpha_beta

    def sample(self, n=None, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.beta(self.alpha, self.beta, n)

    cdef double _summarize(self, double* items, double* weights, int n,
                           int column_idx, int d) nogil:
        cdef int i, j
        cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
        cdef double item

        for i in range(n):
            item = items[i*d + column_idx]
            if isnan(item):
                continue

            w_sum += weights[i]
            x_sum += weights[i] * item
            x2_sum += weights[i] * item * item

            with gil:
                self.summaries[0] += w_sum
                self.summaries[1] += x_sum
                self.summaries[2] += x2_sum

    def from_summaries(self, inertia=0.0):
        # If no summaries stored or the summary is frozen, don't do anything.
        if self.summaries[0] < 1e-8 or self.frozen == True:
                return

        mu = self.summaries[1] / self.summaries[0]
        var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

        if var <= self.min_std:
                self.alpha = 1
                self.beta = 1
        else:
                self.alpha = self.alpha * inertia + ((mu ** 2.0 * (1-mu)) / var - mu) * (1 - inertia)
                self.beta = self.beta * inertia + (((1 - mu) ** 2.0 * mu) / var - (1 - mu)) * (1 - inertia)
        self.summaries = [0, 0, 0]
        self.one_minus_alpha = 1 - self.alpha
        self.one_minus_beta = 1 - self.beta
        self.lgamma_alpha_beta = lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta)

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""

        self.summaries = [0, 0, 0]

    @classmethod
    def blank(cls):
        return cls(0, 1)
