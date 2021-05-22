#!python
#cython: boundscheck=False
#cython: cdivision=True
# PolyExpBetaNormal.pyx
# Contact: Pierre Louvart <pierre.louvart@gmail.com>

import numpy
import scipy
cimport cython

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memset

from scipy.linalg.cython_blas cimport dgemm

from ..utils cimport _log
from ..utils cimport mdot
from ..utils cimport ndarray_wrap_cpointer
from ..utils cimport _is_gpu_enabled
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp
from libc.math cimport fmin as cmin
from libc.math cimport fmax as cmax
from libc.math cimport fabs as cabs
from libc.math cimport pow as cpow
from libc.math cimport isfinite as cisfinite



# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641

eps = 1e-8

cdef class PolyExpBetaNormal(MultivariateDistribution):
	property parameters:
		def __get__(self):
			return [self.alpha, self.beta,
                                self.acoeffs.tolist(), self.bcoeffs.tolist(), self.ccoeffs.tolist(),
                                self.sigmas.tolist()]
		def __set__(self, parameters):
			self.alpha = parameters[0]
                        self.beta = parameters[1]
                        self.acoeffs = np.array(parameters[2])
                        self.bcoeffs = np.array(parameters[3])
                        self.ccoeffs = np.array(parameters[4])
                        self.sigmas = np.array(parameters[5])

	def __cinit__(self, alpha=1.0, beta=1.0, acoeffs=[], bcoeffs=[], ccoeffs=[], sigmas=[], self.frozen=False):
		"""
		Take in the mean vector and the covariance matrix.
		"""
                self.name = "PolyExpBetaNormal"
                self.frozen = frozen
                self.d = 2

                self.alpha = alpha
                self.beta = beta
                self.acoeffs = np.array(acoeffs)
                self.bcoeffs = np.array(bcoeffs)
                self.ccoeffs = np.array(ccoeffs)
                self.sigmas = np.array(sigmas)

                self._alpha = alpha
                self._beta = beta
                self._acoeffs = <double*> self.acoeffs.data
                self._bcoeffs = <double*> self.bcoeffs.data
                self._ccoeffs = <double*> self.ccoeffs.data
                self._sigmas = <double*> self.sigmas.data

		# self.name = "PolyExpBetaNormal"
		# self.frozen = frozen
		# self.mu = numpy.array(means, dtype='float64')
		# self._mu = <double*> self.mu.data
		# self.cov = numpy.array(covariance, dtype='float64')
		# self._cov = <double*> self.cov.data
		# _, self._log_det = numpy.linalg.slogdet(self.cov)

		# if self.mu.shape[0] != self.cov.shape[0]:
			# raise ValueError("mu shape is {} while covariance shape is {}".format(self.mu.shape[0], self.cov.shape[0]))
		# if self.cov.shape[0] != self.cov.shape[1]:
			# raise ValueError("covariance is not a square matrix, dimensions are ({}, {})".format(self.cov.shape[0], self.cov.shape[1]))
		# if self._log_det == NEGINF:
			# raise ValueError("covariance matrix is not invertible.")

		# d = self.mu.shape[0]
		# self.d = d
		# self._inv_dot_mu = <double*> calloc(d, sizeof(double))
		# self._mu_new = <double*> calloc(d, sizeof(double))

		# chol = scipy.linalg.cholesky(self.cov, lower=True)
		# self.inv_cov = scipy.linalg.solve_triangular(chol, numpy.eye(d), lower=True).T
		# self._inv_cov = <double*> self.inv_cov.data
		# mdot(self._mu, self._inv_cov, self._inv_dot_mu, 1, d, d)

		# self.column_sum = <double*> calloc(d*d, sizeof(double))
		# self.column_w_sum = <double*> calloc(d, sizeof(double))
		# self.pair_sum = <double*> calloc(d*d, sizeof(double))
		# self.pair_w_sum = <double*> calloc(d*d, sizeof(double))

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.alpha, self.beta, self.acoeffs, self.bcoeffs, self.ccoeffs, self.sigmas, self.frozen)

	def __dealloc__(self):
                pass
		# free(self._inv_dot_mu)
		# free(self._mu_new)
		# free(self.column_sum)
		# free(self.column_w_sum)
		# free(self.pair_sum)
		# free(self.pair_w_sum)

	cdef void _log_probability(self, double* X, double* logp, int n) nogil:
		cdef int i, j, d = self.d

		for i in range(n):
			# if isnan(X[i*d + j]):
                        # TODO implement nan handling
			logp[i] = X[i*d] ** (self._alpha - 1) + (1 - X[i*d]) ** (1 - self._beta) + lgamma(self._alpha + self._beta) - lgamma(self._alpha) - lgamma(self._beta)
                        for j in range(1, d):
                            logp[i] += self.log_sigma_sqrt_2_pi - ((X[i*d+j] - (self._acoeffs[j] + self._bcoeffs[j] * cexp(-self._ccoeffs[j] * X[i*d+j]))) ** 2) * self.two_sigma_squared

	# cdef double _log_probability_missing(self, double* X) nogil: TODO implement nan handling
		# cdef double logp

		# with gil:
			# X_ndarray = ndarray_wrap_cpointer(X, self.d)
			# avail = ~numpy.isnan(X_ndarray)
			# if avail.sum() == 0:
				# return 0

			# a = numpy.ix_(avail, avail)


			# d1 = PolyExpBetaNormal(self.mu[avail], self.cov[a])
			# logp = d1.log_probability(X_ndarray[avail])

		# return logp

	# def sample(self, n=None, random_state=None): TODO implement random polyexp-beta-normal sampling
		# random_state = check_random_state(random_state)
		# return random_state.multivariate_normal(self.parameters[0],
			# self.parameters[1], n)

        cdef void compute_bins(int bins[nb_bins][2], double * weights, int n) nogil:
            cdef double wsum = 0
            
            cdef int i
            for i in range(n):
                wsum += weights[i]

            cdef double bin_size = wsum / nb_bins

            cdef int pos[nb_bins]
            cdef int s = 0
            cdef int val
            for i in range(nb_bins):
                val = (i+1)**2
                pos[i] = val
                s += val
            cdef double cumsum = 0
            for i in range(nb_bins):
                cumsum += pos[i] * wsum / s
                pos[i] = int(cumsum)

            cdef double cumw = 0
            cdef int curind = 0
            cdef double curval = pos[curind]
            cdef int offset = 0
            bins[0][0] = 0
            for i in range(n):
                cumw += weights[i]
                if cumw > curval:
                    offset = cmax(i, bins[curind][0] + 1)
                    bins[curind][1] = offset
                    curind += 1
                    curval = pos[curind]
                    bins[curind][0] = offset
                    if curind + 1 >= nb_bins:
                        break
            bins[nb_bins-1][1] = n

            return

        cdef void compute_bins_mean(int bins[nb_bins][2], double * res, double * x_view, double * weights, int n, int d, int di) nogil:
            cdef int bl = 0
            cdef int bh = 0
            cdef double cumr = 0
            cdef double accu = 0

            for i in range(nb_bins):
                bl = bins[i][0]
                bh = bins[i][1]
                cumr = 0
                accu = 0
                for j in range(bl, bh):
                    accu += x_view[j*d+di] * weights[j]  
                    cumr += weights[j]
                res[i] = accu / cumr
            
            return

        cdef void compute_bins_sum_ones(int bins[nb_bins][2], double * res, double * x_view, int n, int d, int di) nogil:
            cdef int bl = 0
            cdef int bh = 0
            cdef double accu = 0

            for i in range(nb_bins):
                bl = bins[i][0]
                bh = bins[i][1]
                accu = 0
                for j in range(bl, bh):
                    accu += x_view[j*d+di]
                res[i] = accu
            
            return
                
        cdef void compute_bins_sum(int bins[nb_bins][2], double * res, double * x_view, double * weights, int n, int d, int di) nogil:
            cdef int bl = 0
            cdef int bh = 0
            cdef double accu = 0

            for i in range(nb_bins):
                bl = bins[i][0]
                bh = bins[i][1]
                accu = 0
                for j in range(bl, bh):
                    accu += x_view[j*d+di] * weights[j]  
                res[i] = accu
            
            return
                
        cdef double compute_f(double p[3], Wcparams * wcparams) nogil:
            cdef double a = p[0]
            cdef double b = p[1]
            cdef double c = p[2]

            cdef double r1 = 0
            cdef double r2 = 0
            cdef double r3 = 0
            cdef double e = 0
            for i in range(nb_bins):
                e = cexp(-c * wcparams.xms[i])
                r1 += wcparams.sws[i] * e**2
                r2 += wcparams.sws[i] * e
                r3 += wcparams.sys[i] * e
            return r1 * b**2 + r2 * 2 * a * b - r3 * 2 * b + wcparams.sw * a**2 - 2 * a * wcparams.sy + wcparams.sy2

        cdef void compute_gf(double p[3], Wcparams * wcparams, double res[3]) nogil:
            cdef double a = p[0]
            cdef double b = p[1]
            cdef double c = p[2]

            cdef double ra1 = 0
            cdef double rb1 = 0
            cdef double rb2 = 0
            cdef double rb3 = 0
            cdef double rc1 = 0
            cdef double rc2 = 0
            cdef double rc3 = 0
            cdef double e = 0
            cdef double tmp1 = 0
            cdef double tmp2 = 0
            cdef double tmp3 = 0
            cdef double xmsi
            cdef double swsi

            for i in range(nb_bins):
                xmsi = wcparams.xms[i]
                swsi = wcparams.sws[i]
                e = cexp(-c * xmsi)
                tmp1 = swsi * e
                tmp2 = swsi * e**2
                tmp3 = wcparams.sys[i] * e

                ra1 += tmp1
                rb1 += tmp2
                rb2 += tmp1
                rb3 += tmp3
                rc1 += -2 * xmsi * tmp2
                rc2 += -xmsi * tmp1
                rc3 += -xmsi * tmp3

            res[0] = wcparams.sw*2*a - 2*wcparams.sy + 2*b*ra1
            res[1] = b*2*rb1 + 2*a*rb2 - 2*rb3
            res[2] = b**2*rc1 + 2*a*b*rc2 - 2*b*rc3

            return


        cdef (double, double, double, double, double, double, double, int) dcstep(double stx, double fx, double dx, double sty, double fy, double dy, double stp, double fp, double dp, int brackt, double stpmin, double stpmax) nogil:
            cdef double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta

            sgnd = dp*(dx/cabs((dx))

            #   First case: A higher function value. The minimum is bracketed.
            #   If the cubic step is closer to stx than the quadratic step, the
            #   cubic step is taken, otherwise the average of the cubic and
            #   quadratic steps is taken.
            if fp > fx:
                theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
                s = fmax(fmax(cabs((theta),cabs((dx)),cabs((dp))
                gamma = s*csqrt(cpow((theta/s),2)-(dx/s)*(dp/s))
                if stp < stx:
                    gamma = -gamma
                p = (gamma-dx) + theta
                q = ((gamma-dx)+gamma) + dp
                r = p/q
                stpc = stx + r*(stp-stx)
                stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2.0)*(stp-stx)
                if cabs((stpc-stx) < cabs((stpq-stx):
                    stpf = stpc
                else:
                    stpf = stpc + (stpq-stpc)/2.0
                brackt = 1

            #   Second case: A lower function value and derivatives of opposite
            #   sign. The minimum is bracketed. If the cubic step is farther from
            #   stp than the secant step, the cubic step is taken, otherwise the
            #   secant step is taken.
            elif sgnd < 0.0:
                theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
                s = fmax(fmax(cabs((theta),cabs((dx)),cabs((dp))
                gamma = s*csqrt((theta/s)**2-(dx/s)*(dp/s))
                if stp > stx:
                    gamma = -gamma
                p = (gamma-dp) + theta
                q = ((gamma-dp)+gamma) + dx
                r = p/q
                stpc = stp + r*(stx-stp)
                stpq = stp + (dp/(dp-dx))*(stx-stp)
                if cabs((stpc-stp) > cabs((stpq-stp):
                    stpf = stpc
                else:
                    stpf = stpq
                brackt = 1

            #   Third case: A lower function value, derivatives of the same sign,
            #   and the magnitude of the derivative decreases.
            elif cabs((dp) < cabs((dx):
                #	The cubic step is computed only if the cubic tends to infinity
                #	in the direction of the step or if the minimum of the cubic
                #	is beyond stp. Otherwise the cubic step is defined to be the
                #	secant step.
                theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
                s = fmax(fmax(cabs((theta),cabs((dx)),cabs((dp))

                #	The case gamma = 0 only arises if the cubic does not tend
                #	to infinity in the direction of the step.
                gamma = s*csqrt(fmax(0.0,cpow((theta/s),2)-(dx/s)*(dp/s)))
                if stp > stx:
                    gamma = -gamma
                p = (gamma-dp) + theta
                q = (gamma+(dx-dp)) + gamma
                r = p/q
                if r < 0.0 and gamma != 0.0:
                    stpc = stp + r*(stx-stp)
                elif stp > stx:
                    stpc = stpmax
                else:
                    stpc = stpmin
                stpq = stp + (dp/(dp-dx))*(stx-stp)

                if brackt:
                    #	   A minimizer has been bracketed. If the cubic step is
                    #	   closer to stp than the secant step, the cubic step is
                    #	   taken, otherwise the secant step is taken.
                    if cabs((stpc-stp) < cabs((stpq-stp):
                       stpf = stpc
                    else:
                       stpf = stpq
                    if stp > stx:
                       stpf = cmin(stp+0.66*(sty-stp),stpf)
                    else:
                       stpf = cmax(stp+0.66*(sty-stp),stpf)
                else:
                    #	   A minimizer has not been bracketed. If the cubic step is
                    #	   farther from stp than the secant step, the cubic step is
                    #	   taken, otherwise the secant step is taken.
                    if cabs((stpc-stp) > cabs((stpq-stp):
                       stpf = stpc
                    else:
                       stpf = stpq
                    stpf = cmin(stpmax,stpf)
                    stpf = cmax(stpmin,stpf)

            #   Fourth case: A lower function value, derivatives of the same sign,
            #   and the magnitude of the derivative does not decrease. If the
            #   minimum is not bracketed, the step is either stpmin or stpmax,
            #   otherwise the cubic step is taken.
            else:
                if brackt:
                    theta = 3.0*(fp-fy)/(sty-stp) + dy + dp
                    s = fmax(fmax(cabs((theta),cabs((dy)),cabs((dp))
                    gamma = s*csqrt((theta/s)**2-(dy/s)*(dp/s))
                    if stp > sty:
                        gamma = -gamma
                    p = (gamma-dp) + theta
                    q = ((gamma-dp)+gamma) + dy
                    r = p/q
                    stpc = stp + r*(sty-stp)
                    stpf = stpc
                elif stp > stx:
                    stpf = stpmax
                else:
                    stpf = stpmin

            #   Update the interval which contains a minimizer.
            if fp > fx:
                sty = stp
                fy = fp
                dy = dp
            else:
                if sgnd < 0.0:
                     sty = stx
                     fy = fx
                     dy = dx
                stx = stp
                fx = fp
                dx = dp

            return stx, fx, dx, sty, fy, dy, stpf, brackt


        cdef double dot(double gfk[3], double pk[3]) nogil:
            return gfk[0] * pk[0] + gfk[1] * pk[1] + gfk[2] * pk[2]

        cdef (double, double, double) line_search_wolfe1(double xk[3], double pk[3], double gfk[3], double gfkp1[3], double old_fval, double old_old_fval, Wcparams * wcparams) nogil:
            cdef double gfval = dot(gfk, pk)
            cdef double fval = old_fval
            cdef double stp

            if gfval != 0:
                stp = cmin(1.0, 1.01 * 2 * (old_fval - old_old_fval) / gfval)
                if stp < 0:
                    stp = 1.0
            else:
                stp = 1.0

            cdef double p5 = 0.5
            cdef double xtrapl = 1.1
            cdef double xtrapu = 4.0

            cdef double ftest, fm, fxm, fym, gm, gxm, gym
            cdef double ftol = 1e-4
            cdef double gtol = 0.9
            cdef double xtol = 1e-14
            cdef double stpmin = 1e-100
            cdef double stpmax = 1e100

            if stp < stpmin: return 0, 0, 0
            if stp > stpmax: return 0, 0, 0
            if gfval >= 0.0: return 0, 0, 0

            #      Initialize local variables.
            cdef int brackt = 0
            cdef int stage = 1
            cdef double finit = fval
            cdef double ginit = gfval
            cdef double gtest = ftol * ginit
            cdef double width = stpmax - stpmin
            cdef double width1 = width / p5
            cdef double stx = 0.0
            cdef double fx = finit
            cdef double gx = ginit
            cdef double sty = 0.0
            cdef double fy = finit
            cdef double gy = ginit
            cdef double stmin = 0.0
            cdef double stmax = stp + xtrapu * stp

            cdef double xktmp[3]

            for i in range(100):
                # print(i)
                xktmp[0] = xk[0] + stp * pk[0]; xktmp[1] = xk[1] + stp * pk[1]; xktmp[2] = xk[2] + stp * pk[2]; 
                fval = compute_f(xktmp, wcparams)
                compute_gf(xktmp, wcparams, gfkp1)
                gfval = dot(gfkp1, pk)
                # print("START", brackt, stage, ginit,gtest,gx,gy,finit,fx,fy,stx,sty,stmin,stmax,width,width1)
                # print("PRE-DEBUG 1", stp, fval, gfval, ftol, gtol, xtol, stpmin, stpmax)
                #   If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
                #   algorithm enters the second stage.
                ftest = finit + stp * gtest
                if stage == 1 and fval <= ftest and gfval >= 0.0:
                    stage = 2

                #   Test for warnings.
                if brackt and (stp <= stmin or stp >= stmax):
                    return 0, 0, 0
                if brackt and stmax - stmin <= xtol * stmax:
                    return 0, 0, 0
                if stp == stpmax and fval <= ftest and gfval <= gtest:
                    return 0, 0, 0
                if stp == stpmin and (fval > ftest or gfval >= gtest):
                    return 0, 0, 0

                #   Test for convergence.
                if fval <= ftest and cabs((gfval) <= gtol * (-ginit):
                    break

                #   A modified function is used to predict the step during the
                #   first stage if a lower function value has been obtained but
                #   the decrease is not sufficient.
                if stage == 1 and fval <= fx and fval > ftest:
                    #      Define the modified function and derivative values.
                    fm = fval - stp * gtest
                    fxm = fx - stx * gtest
                    fym = fy - sty * gtest
                    gm = gfval - gtest
                    gxm = gx - gtest
                    gym = gy - gtest

                    #      Call dcstep to update stx, sty, and to compute the new step.
                    stx, fxm, gxm, sty, fym, gym, stp, brackt = dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

                    #      Reset the function and derivative values for f.
                    fx = fxm + stx*gtest
                    fy = fym + sty*gtest
                    gx = gxm + gtest
                    gy = gym + gtest
                else:
                    #     Call dcstep to update stx, sty, and to compute the new step.
                    stx, fx, gx, sty, fy, gy, stp, brackt = dcstep(stx, fx, gx, sty, fy, gy, stp, fval, gfval, brackt, stmin, stmax)

                #   Decide if a bisection step is needed.
                if brackt:
                    if (cabs((sty-stx) >= 0.66 * width1):
                        stp = stx + p5 * (sty - stx)
                    width1 = width
                    width = cabs((sty-stx)

                #   Set the minimum and maximum steps allowed for stp.
                if brackt:
                    stmin = cmin(stx,sty)
                    stmax = cmax(stx,sty)
                else:
                    stmin = stp + xtrapl * (stp-stx)
                    stmax = stp + xtrapu * (stp-stx)

                #   Force the step to be within the bounds stpmax and stpmin.
                stp = cmin(cmax(stp,stpmin),stpmax)

                #   If further progress is not possible, let stp be the best
                #   point obtained during the search.
                if brackt and (stp <= stmin or stp >= stmax) or (brackt and stmax-stmin <= xtol * stmax):
                    stp = stx

            else:
                return 1, 0, 0

            return 0, stp, fval

        cdef double vecnorm(double x[3]) nogil:
            return fmax(fmax(cabs((x[0]), cabs((x[1])), cabs((x[2]))

        cdef void dotm(double m1[3][3], double m2[3][3], double res[3][3]) nogil:
            cdef double accu
            cdef int i, j, k
            for i in range(3):
                for j in range(3):
                    accu = 0
                    for k in range(3):
                        accu += m1[i][k] * m2[k][j]
                    res[i][j] = accu

        cdef void dotmv(double m1[3][3], double v[3], double res[3]) nogil:
            cdef double accu
            cdef int i, k
            for i in range(3):
                accu = 0
                for k in range(3):
                    accu += m1[i][k] * v[k]
                res[i] = accu

        cdef double norm(double x[3]) nogil:
            return csqrt(x[0]**2 + x[1]**2 + x[2]**2)

        cdef void compute_polyexp_coeffs(WCparams * wcparams, double a_prior, double b_prior, double c_prior, double res_view[3]) nogil:
            cdef int maxiter = 600
            cdef double gtol = 1e-5
            
            cdef double xk[3] # TODO xk unnecessary in regards to res_view
            xk[0] = a_prior; xk[1] = b_prior; xk[2] = c_prior
            cdef double old_fval = compute_f(xk, wcparams)
            cdef double gfk[3]
            compute_gf(xk, wcparams, gfk)
            
            cdef double k = 0
            cdef double I[3][3]
            for i in range(3):
                for j in range(3):
                    I[i][j] = i == j
            cdef double Hk[3][3]
            for i in range(3):
                for j in range(3):
                    Hk[i][j] = i == j
            cdef double Hktmp[3][3]


            cdef double old_old_fval = old_fval + norm(gfk) / 2

            # BEGIN NEW VARS
            cdef double pk[3]
            cdef double xkp1[3]
            cdef double gfkp1[3]
            cdef double sk[3]
            cdef double yk[3]
            cdef double A1[3][3]
            cdef double A2[3][3]
            cdef double alpha_k = 0
            cdef double rhok = 0
            cdef double rhok_inv = 0
            cdef double tmp_fval = 0
            # END NEW VARS

            cdef double gnorm = vecnorm(gfk)
            while (gnorm > gtol) and (k < maxiter):
                dotmv(Hk, gfk, pk)
                pk[0] = -pk[0]; pk[1] = -pk[1]; pk[2] = -pk[2]
                # print(xk, pk, gfk)
                tmp_fval = old_fval
                status, alpha_k, old_fval = line_search_wolfe1(xk, pk, gfk, gfkp1, old_fval, old_old_fval, wcparams)
                old_old_fval = tmp_fval
                if status:
                    break

                xkp1[0] = xk[0] + alpha_k * pk[0]; xkp1[1] = xk[1] + alpha_k * pk[1]; xkp1[2] = xk[2] + alpha_k * pk[2]
                sk[0] = xkp1[0] - xk[0]; sk[1] = xkp1[1] - xk[1]; sk[2] = xkp1[2] - xk[2]
                xk[0] = xkp1[0]; xk[1] = xkp1[1]; xk[2] = xkp1[2]

                yk[0] = gfkp1[0] - gfk[0]; yk[1] = gfkp1[1] - gfk[1]; yk[2] = gfkp1[2] - gfk[2]
                gfk[0] = gfkp1[0]; gfk[1] = gfkp1[1]; gfk[2] = gfkp1[2]
                k += 1
                gnorm = vecnorm(gfk)
                if (gnorm <= gtol):
                    break

                if not cisfinite(old_fval):
                    break

                rhok_inv = dot(yk, sk)
                # this was handled in numeric, let it remaines for more safety
                if rhok_inv == 0.:
                    rhok = 1000.0
                    #print("Divide-by-zero encountered: rhok assumed large")
                else:
                    rhok = 1. / rhok_inv

                A1[0][0] = 1 - sk[0] * yk[0] * rhok
                A1[1][1] = 1 - sk[1] * yk[1] * rhok
                A1[2][2] = 1 - sk[2] * yk[2] * rhok
                A1[0][1] = - sk[0] * yk[1] * rhok
                A1[0][2] = - sk[0] * yk[2] * rhok
                A1[1][0] = - sk[1] * yk[0] * rhok
                A1[1][2] = - sk[1] * yk[2] * rhok
                A1[2][0] = - sk[2] * yk[0] * rhok
                A1[2][1] = - sk[2] * yk[1] * rhok
                A2[0][0] = 1 - yk[0] * sk[0] * rhok
                A2[1][1] = 1 - yk[1] * sk[1] * rhok
                A2[2][2] = 1 - yk[2] * sk[2] * rhok
                A2[0][1] = - yk[0] * sk[1] * rhok
                A2[0][2] = - yk[0] * sk[2] * rhok
                A2[1][0] = - yk[1] * sk[0] * rhok
                A2[1][2] = - yk[1] * sk[2] * rhok
                A2[2][0] = - yk[2] * sk[0] * rhok
                A2[2][1] = - yk[2] * sk[1] * rhok
                dotm(Hk, A2, Hktmp)
                dotm(A1, Hktmp, Hk)
                Hk[0][0] += sk[0] * sk[0] * rhok
                Hk[1][1] += sk[1] * sk[1] * rhok
                Hk[2][2] += sk[2] * sk[2] * rhok
                Hk[0][1] += sk[0] * sk[1] * rhok
                Hk[0][2] += sk[0] * sk[2] * rhok
                Hk[1][0] += sk[1] * sk[0] * rhok
                Hk[1][2] += sk[1] * sk[2] * rhok
                Hk[2][0] += sk[2] * sk[0] * rhok
                Hk[2][1] += sk[2] * sk[1] * rhok

            res_view[0] = xk[0]
            res_view[1] = xk[1]
            res_view[2] = xk[2]
        

	cdef double _summarize(self, double* X, double* weights, int n,
		int column_idx, int d) nogil:
		"""Calculate sufficient statistics for a minibatch.

		The sufficient statistics for a multivariate gaussian update is the sum of
		each column, and the sum of the outer products of the vectors.
		"""
            cdef double * polyexp_coeffs = <double*> calloc(d * 3, sizeof(double))
            cdef double * sigmas = <double*> calloc(d, sizeof(double))
            cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
            cdef double item

            cdef int bins[nb_bins][2]
            cdef Wcparams wcparams
            cdef double accu_sw = 0
            cdef double accu_sy = 0
            cdef double accu_sy2 = 0
            cdef double min_y
            cdef double max_y
            cdef double accu_tmp
            cdef double accu_x
            cdef double mu, var
            cdef int i

            compute_bins(bins, weights, n)

            for di in range(d):
                compute_bins_sum_ones(bins, wcparams.sws, weights, n, d, di)
                compute_bins_sum(bins, wcparams.sys, y_view, weights, n, d, di)
                compute_bins_mean(bins, wcparams.xms, x_view, weights, n, d, di)
                accu_sw = 0
                accu_sy = 0
                accu_sy2 = 0
                min_y = 1
                max_y = 0
                for i in range(n):
                    accu_x = X[i * d + di]
                    min_y = cmin(min_y, accu_x)
                    max_y = cmax(max_y, accu_x)
                    accu_tmp = accu_x * weights[i]
                    accu_sw += weights[i]
                    accu_sy += accu_tmp
                    accu_sy2 += accu_tmp * accu_tmp
                wcparams.sw = accu_sw
                wcparams.sy = accu_sy
                wcparams.sy2 = accu_sy2

                sigmas[di] = accu_sy2 / accu_sw - accu_sy ** 2.0 / accu_sw ** 2.0
                compute_polyexp_coeffs(&wcparams, min_y, max_y - min_y, 25.0, polyexp_coeffs + i * 3)

            for i in range(n):
                item = X[i * d]
                w_sum += weights[i]
                x_sum += item * weights[i]
                x2_sum += item * item * weights[i]
            mu = x_sum / w_sum
            var = x2_sum / w_sum - x_sum ** 2.0 / w_sum ** 2.0

            with gil:
                for di in range(d):
                    self._acoeffs[d] = polyexp_coeffs[d*3]
                    self._bcoeffs[d] = polyexp_coeffs[d*3+1]
                    self._ccoeffs[d] = polyexp_coeffs[d*3+2]
                self._sigmas = sigmas
                self._alpha = self.alpha * ((mu ** 2.0 * (1-mu)) / var - mu)
                self._beta = self.beta * (((1 - mu) ** 2.0 * mu) / var - (1 - mu))
                self.alpha = self._alpha
                self.beta = self._beta
                
            free(sigmas)
            free(polyexp_coeffs)

	def from_summaries(self, inertia=0.0, min_covar=1e-5):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		cdef int d = self.d, i, j, k

		# If no summaries stored or the summary is frozen, don't do anything.
                if self.frozen == True:# or w_sum < 1e-7:
			return

		self.clear_summaries()

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		memset(self.sigmas, 0, self.d*sizeof(double))
		memset(self.acoeffs, 0, self.d*sizeof(double))
		memset(self.bcoeffs, 0, self.d*sizeof(double))
		memset(self.ccoeffs, 0, self.d*sizeof(double))
                self.alpha = 1.0
                self.beta = 1.0
                self._alpha = 1.0
                self._beta = 1.0

	@classmethod
	def from_samples(cls, X, weights=None, **kwargs):
		"""Fit a distribution to some data without pre-specifying it."""

		distribution = cls.blank(X.shape[1])
		distribution.fit(X, weights, **kwargs)
		return distribution

	@classmethod
	def blank(cls, d=2):
		sigmas = numpy.zeros(d)
		acoeffs = numpy.zeros(d)
		bcoeffs = numpy.zeros(d)
		ccoeffs = numpy.zeros(d)
                alpha = 1.0
                beta = 1.0
		return cls(alpha, beta, acoeffs, bcoeffs, ccoeffs, sigmas)
