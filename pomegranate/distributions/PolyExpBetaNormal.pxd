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
    cdef double* _bounds
    cdef double* two_sigma_squared
    cdef double* log_sigma_sqrt_2_pi
    cdef void _log_probability(self, double* X, double* logp, int n) nogil
    
    cdef void compute_bins(self, int bins[nb_bins][2], double * weights, int n) nogil
    cdef void compute_bins_mean(self, int bins[nb_bins][2], double * res, double * x_view, double * weights, int n, int d, int di) nogil
    cdef void compute_bins_sum_ones(self, int bins[nb_bins][2], double * res, double * x_view, int n, int d, int di) nogil
    cdef void compute_bins_sum(self, int bins[nb_bins][2], double * res, double * x_view, double * weights, int n, int d, int di) nogil
    cdef double compute_f(self, double p[3], Wcparams * wcparams) nogil
    cdef void compute_gf(self, double p[3], Wcparams * wcparams, double res[3]) nogil
    cdef (double, double, double, double, double, double, double, int) dcstep(self, double stx, double fx, double dx, double sty, double fy, double dy, double stp, double fp, double dp, int brackt, double stpmin, double stpmax) nogil
    cdef double dot(self, double gfk[3], double pk[3]) nogil
    cdef (double, double, double) line_search_wolfe1(self, double xk[3], double pk[3], double gfk[3], double gfkp1[3], double old_fval, double old_old_fval, Wcparams * wcparams) nogil
    cdef double vecnorm(self, double x[3]) nogil
    cdef void dotm(self, double m1[3][3], double m2[3][3], double res[3][3]) nogil
    cdef void dotmv(self, double m1[3][3], double v[3], double res[3]) nogil
    cdef double norm(self, double x[3]) nogil
    cdef void compute_polyexp_coeffs(self, Wcparams * wcparams, double a_prior, double b_prior, double c_prior, double res_view[3], double a_min, double a_max, double b_min, double b_max, double c_min, double c_max) nogil
    







