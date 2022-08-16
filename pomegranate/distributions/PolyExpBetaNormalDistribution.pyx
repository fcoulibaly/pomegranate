#!python
#cython: boundscheck=False
#cython: cdivision=True
# PolyExpBetaNormalDistribution.pyx
# Contact: Pierre Louvart <pierre.louvart@gmail.com>

import numpy
import scipy
import matplotlib.pyplot as plt

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.stdio cimport printf

from scipy.linalg.cython_blas cimport dgemm

from ..utils cimport _log
from ..utils cimport lgamma
from ..utils cimport ndarray_wrap_cpointer
from ..utils cimport _is_gpu_enabled
from ..utils cimport isnan
from ..utils import check_random_state

from ..regression.polyexp cimport polyexp_fit
# from ..regression.polyexp cimport polyexp_fit

from libc.math cimport sqrt as csqrt
from libc.math cimport pow as cpow
from libc.math cimport exp as cexp

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

eps = 1e-8

cdef class PolyExpBetaNormalDistribution(MultivariateDistribution):
    property parameters:
        def __get__(self):
            return [self.coeffs.tolist(), self.sigmas.tolist(), self._alpha, self._beta, self.lower_bounds.tolist(), self.upper_bounds.tolist()]
        def __set__(self, parameters):
            self.coeffs = numpy.array(parameters[0])
            self.sigmas = numpy.array(parameters[1])
            self._alpha = parameters[2]
            self._beta = parameters[3]
            self.lower_bounds = numpy.array(parameters[4])
            self.upper_bounds = numpy.array(parameters[5])

    def __cinit__(self, coeffs=[], sigmas=[], alpha=1, beta=1, lower_bounds=[], upper_bounds=[], frozen=False):
        """
            For n (n in N) reflectances rn, and depth x

            coeffs = [[a1, b1, c1], [a2, b2, c2], ..., [an, bn, cn]]
            with an, bn, cn coeffs for reflectance equation yn = an + bn * e^(-cn * x)

            sigma = [s1, s2, ..., sn]
            with sn standard deviation for reflectance rn

            alpha,beta parameters of joint beta distribution along depth
        """

        self.name = "PolyExpBetaNormalDistribution"
        self.frozen = frozen

        self.has_apriori = True
        self.coeffs = numpy.array(coeffs, dtype='float64')
        self.sigmas = numpy.array(sigmas, dtype='float64')
        self._alpha = alpha
        self._beta = beta
        d = self.coeffs.shape[0]
        self.d = d + 1
        if len(lower_bounds) == 0:
            lower_bounds = numpy.ones((d,3)) * numpy.inf
        if len(upper_bounds) == 0:
            upper_bounds = numpy.ones((d,3)) * numpy.inf
        self.lower_bounds = numpy.array(lower_bounds, dtype='float64')
        self.upper_bounds = numpy.array(upper_bounds, dtype='float64')

        if self.coeffs.shape[0] != self.sigmas.shape[0]: # TODO: Bounds shape check
            raise ValueError("coeffs and sigmas arrays length mismatch ({} and {} respectively)".format(self.coeffs.shape[0], self.sigmas.shape[0]))
        if self.coeffs.shape[1] != 3:
            raise ValueError("coeffs list should have 3 subparameters, got {}".format(self.coeffs.shape[1]))

        # Convenience variable
        self.lgamma_constants = lgamma(self._alpha + self._beta) - lgamma(self._alpha) - lgamma(self._beta)
        self.log_sigma_sqrt_2_pi = <double*> calloc(d, sizeof(double))
        self.two_sigma_squared = <double*> calloc(d, sizeof(double))
        for di in range(self.d-1):
            self.log_sigma_sqrt_2_pi[di] = -_log(self.sigmas[di] * SQRT_2_PI)
            self.two_sigma_squared[di] = 1. / (2 * self.sigmas[di] ** 2) if self.sigmas[di] > 0 else 0
        self._coeffs = <double*> self.coeffs.data
        self.acoeffs = self.coeffs[:,0]
        self._acoeffs = <double*> self.acoeffs.data
        self.bcoeffs = self.coeffs[:,1]
        self._bcoeffs = <double*> self.bcoeffs.data
        self.ccoeffs = self.coeffs[:,2]
        self._ccoeffs = <double*> self.ccoeffs.data
        self._sigmas = <double*> self.sigmas.data
        self._lower_bounds = <double*> self.lower_bounds.data
        self._upper_bounds = <double*>self.upper_bounds.data
    
        self.infered_coeffs = numpy.zeros((d,3))
        self.infered_sigmas = numpy.zeros((d,))
        self._infered_coeffs = <double*> self.infered_coeffs.data
        self._infered_sigmas = <double*> self.infered_sigmas.data
        self._infered_alpha = 0
        self._infered_beta = 0

    def __reduce__(self):
        """Serialize the distribution for pickle."""
        return self.__class__, (self.coeffs, self.sigmas, self._alpha, self._beta, self.lower_bounds, self.upper_bounds, self.frozen)

    def __dealloc__(self):
        free(self.log_sigma_sqrt_2_pi)
        free(self.two_sigma_squared)

    cdef void _log_probability(self, double* X, double* logp, int n) nogil:
        cdef int i, j, d = self.d-1
        cdef double x = 0

        if _is_gpu_enabled():
            raise NotImplemented("GPU is not available for BNPE distributions")
        else:
            for i in range(n):
                if X[i*self.d] <= 0 or X[i*self.d] >= 1:
                    logp[i] = 0
                else:
                    # Depth beta-distribution likelyhood
                    x = X[i*self.d]
                    logp[i] = _log(x) * (self._alpha-1) + _log(1 - x) * (self._beta-1) + self.lgamma_constants
                    for j in range(d):
                        # Dimension specific normal distribution likelyhood
                        logp[i] += self.log_sigma_sqrt_2_pi[j] - ((X[i*self.d+j+1] - (self._coeffs[j*3] + self._coeffs[j*3+1] * cexp(-self._coeffs[j*3+2] * x))) ** 2) * self.two_sigma_squared[j]


    cdef double _log_probability_missing(self, double* X) nogil: # TODO not implemented
        cdef double logp

        with gil:
            X_ndarray = ndarray_wrap_cpointer(X, self.d)
            avail = ~numpy.isnan(X_ndarray)
            if avail.sum() == 0:
                return 0

            a = numpy.ix_(avail, avail)


            d1 = PolyExpBetaNormalDistribution(self.mu[avail], self.cov[a])
            logp = d1.log_probability(X_ndarray[avail])

            return logp

    def sample(self, n=None, random_state=None):
        random_state = check_random_state(random_state)
        x = random_state.beta(self._alpha, self._beta)
        y = [self._acoeffs[di] + self._bcoeffs[di] * numpy.exp(-self._ccoeffs[di] * x) + random_state.normal(0, self.sigmas[di], n) for di in range(self.d-1)]
        return numpy.dstack((x, *y))

    cdef double _summarize(self, double* X, double* weights, int n,
                           int column_idx, int d) nogil:
        """Calculate sufficient statistics for a minibatch.

            The sufficient statistics for a multivariate gaussian update is the sum of
            each column, and the sum of the outer products of the vectors.
        """
        
        cdef int i, di
        cdef double a,b,c
        cdef double mu, var, w_sum, x_sum, x2_sum
        cdef double accu_tmp, accu_x, accu_sy, accu_sy2, accu_sw
        cdef double res[3]

        accu_sw = 0.0
        for i in range(n):
            accu_sw += weights[i]
        
        for di in range(self.d-1):
            polyexp_fit(
                X, weights, n, di + 1, self.d,
                &self._coeffs[di*3],
                &self._lower_bounds[di*3],
                &self._upper_bounds[di*3],
                True,
                15,
                res
            )
            a = res[0]
            b = res[1]
            c = res[2]
        
            accu_sy = 0
            accu_sy2 = 0
            for i in range(n):
                accu_x = X[i * self.d + di + 1] - (a + b * cexp(-c * X[i * self.d]))
                accu_tmp = accu_x * weights[i]
                accu_sy += accu_tmp
                accu_sy2 += accu_tmp * accu_x
            if accu_sw == 0:
                self._infered_sigmas[di] = 0
            else:
                self._infered_sigmas[di] = csqrt(accu_sy2 / accu_sw - cpow(accu_sy, 2) / cpow(accu_sw, 2))
            
            self._infered_coeffs[di*3+0] = a
            self._infered_coeffs[di*3+1] = b
            self._infered_coeffs[di*3+2] = c

        w_sum = 0.0
        x_sum = 0.0
        x2_sum = 0.0
        for i in range(n):
            item = X[i * self.d]
            w_sum += weights[i]
            x_sum += item * weights[i]
            x2_sum += item * item * weights[i]

        if w_sum == 0:
            mu = 0
            var = 0
        else:
            mu = x_sum / w_sum
            var = x2_sum / w_sum - x_sum ** 2.0 / w_sum ** 2.0

        if var == 0:
            self._infered_alpha = 1
            self._infered_beta = 1
        else:
            self._infered_alpha = ((mu ** 2.0 * (1-mu)) / var - mu)
            self._infered_beta = (((1 - mu) ** 2.0 * mu) / var - (1 - mu))
        

    def from_summaries(self, inertia=0.0, min_covar=1e-5):
        """
        Set the parameters of this Distribution to maximize the likelihood of
            the given sample. Items holds some sort of sequence. If weights is
            specified, it holds a sequence of value to weight each item by.
        """
        self.coeffs[:] = self.coeffs * inertia + self.infered_coeffs * (1 - inertia)
        self.sigmas[:] = self.sigmas * inertia + self.infered_sigmas * (1 - inertia)
        self._alpha = self._alpha * inertia + self._infered_alpha * (1 - inertia)
        self._beta = self._beta * inertia + self._infered_beta * (1 - inertia)

        self.lgamma_constants = lgamma(self._alpha + self._beta) - lgamma(self._alpha) - lgamma(self._beta)
        for di in range(self.d-1): # TODO update later
            self.log_sigma_sqrt_2_pi[di] = -_log(self.sigmas[di] * SQRT_2_PI) if self.sigmas[di] > 0 else 0
            self.two_sigma_squared[di] = 1. / (2 * self.sigmas[di] ** 2) if self.sigmas[di] > 0 else 0

        self.clear_summaries()

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""
        memset(self._infered_coeffs, 0, 3 * (self.d-1) * sizeof(double))
        memset(self._infered_sigmas, 0, (self.d-1) * sizeof(double))
        self._infered_alpha = 0
        self._infered_beta = 0

    def weighted_quantile(self, values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = numpy.array(values)
        quantiles = numpy.array(quantiles)
        if sample_weight is None:
            sample_weight = numpy.ones(len(values))
        sample_weight = numpy.array(sample_weight)
        assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = numpy.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]

        weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= numpy.sum(sample_weight)
        return numpy.interp(quantiles, weighted_quantiles, values)

    def abc_apriori(self, d, y, weights, k=10):
        """
        Returns a rough apriori for a,b and c params that can be
        used as starting point for L-BFGS-B regression
        """
        # plt.scatter(d, y, c=weights)
        # plt.show()
        if weights is None:
            weights = numpy.ones(len(d), dtype=float)

        if len(d) >= 10_000:
            inds = numpy.random.choice(len(weights), 10_000, p = weights/sum(weights))
            d = d[inds]
            y = y[inds]
            weights = weights[inds]

        # plt.scatter(d, y, c=weights)
        # plt.show()

        n = len(d)

        ori_d = d
        ori_y = y
        
        if n == 0: raise Exception("Error")
        if n == 1: return y[0],0, 0
        if n == 2: return #TODO

        def compute_weights(weights):
            if sum(weights) > 0:
                return weights
            else:
                return None

        if n > 3:
            if n <= 3 * k:
                nb_intervals = 3
            else:
                nb_intervals = k

            d_quantiles = self.weighted_quantile(d, numpy.linspace(0,1,nb_intervals+1), sample_weight=weights)
            d_intervals = list(zip(d_quantiles[:-1],d_quantiles[1:]))

            d_new = numpy.array([numpy.average(d[(first <= d) & (d <= last)], weights=compute_weights(weights[(first <= d) & (d <= last)]) ) for first,last in d_intervals])
            y_new = numpy.array([numpy.average(y[(first <= d) & (d <= last)], weights=compute_weights(weights[(first <= d) & (d <= last)]) ) for first,last in d_intervals])
            d = d_new
            y = y_new
            n = len(d)


        d1,d2,d3 = numpy.quantile(d, [0,.5,1])
        y1,y2,y3 = numpy.interp([d1,d2,d3], d, y)
        
        if (y1 - y2) * (y2 - y3) < 0:
            if abs(y1 - y2) < abs(y3 - y2):
                y1,y2 = y2,y1
            else:
                y3,y2 = y2,y3

        def optimization_func(c):
            return (numpy.exp(-c * d1) - numpy.exp(-c * d3)) / (numpy.exp(-c * d1) - numpy.exp(-c * d2)) - (y1 - y3) / (y1 - y2)

        def bisection(cmin, cmax, it):
            c = (cmin + cmax) / 2
            if it == 0: return c
            s1 = optimization_func(cmin)
            s2 = optimization_func(cmax)
            if s1 * s2 > 0:
                return bisection(cmax,cmax*2, it-1)
            if optimization_func(c) * s1 > 0:
                return bisection(c, cmax, it-1)
            return bisection(cmin, c, it-1)

        if abs((y2-y1) / (d2 - d1)) > abs((y3-y2) / (d3 - d2)):
            cmin,cmax = 0.001,30.
        else:
            cmin,cmax = -30.,-0.001

        c = bisection(cmin,cmax,10)
        tmp = (numpy.exp(-c * d2) - numpy.exp(-c * d1))
        if tmp != 0:
            b = (y2-y1) / tmp
        else:
            b = 0
        a = y1 - b * numpy.exp(-c * d1)

        return a,b,c

    @classmethod
    def from_samples(cls, X, weights=None, lower_bounds=[], upper_bounds=[], **kwargs):
        """Fit a distribution to some data without pre-specifying it."""

        # Dimension minus 1 because the first is the beta-based dimension
        distribution = cls.blank(X.shape[1], lower_bounds=lower_bounds, upper_bounds=upper_bounds, **kwargs)

        if "coeffs" not in kwargs:
            for di in range(1, X.shape[1]):
                apriori = distribution.abc_apriori(X[:,0], X[:,di], weights)
                lower = numpy.where(distribution.lower_bounds[di-1] == numpy.inf, -numpy.inf, distribution.lower_bounds[di-1])
                upper = distribution.upper_bounds[di-1]
                distribution.coeffs[di-1,:] = numpy.clip(lower, apriori, upper)
        distribution.fit(X, weights)
        # for di in range(1, X.shape[1]):
        return distribution

    @classmethod
    def blank(cls, d=2, coeffs=None, sigmas=None, alpha=None, beta=None, lower_bounds=[], upper_bounds=[]):
        if coeffs is None:
            coeffs = numpy.zeros((d-1, 3))
        if sigmas is None:
            sigmas = numpy.zeros(d-1)
        if alpha is None:
            alpha = 1.
        if beta is None:
            beta = 1.
        return cls(coeffs, sigmas, alpha, beta, lower_bounds, upper_bounds)
