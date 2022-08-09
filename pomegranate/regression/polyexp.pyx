cimport numpy as np
import numpy as np
import cython
from typing import Optional
from libc.stdlib cimport malloc
from libc.stdio cimport printf
from libc.math cimport exp

cdef extern from "math.h":
    float INFINITY

cdef extern from "./lbfgsb.h":
    cdef int setulb(long int *n, long int *m, double *x, 
	double *l, double *u, long int *nbd, double *f, double 
	*g, double *factr, double *pgtol, double *wa, long int *
	iwa, long int *task, long int *iprint, long int *csave, long int *lsave, 
	long int *isave, double *dsave) nogil
    cdef int START
    cdef int FG
    cdef int FG_END
    cdef int NEW_X

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef int polyexp_fit(
    double * X,
    double * w,
    int nb_samples,
    int di, # dimension to use for y
    int d, # number of dimensions
    double * p0,
    double * lower_bounds,
    double * upper_bounds,
    int assume_sorted,
    int nb_quantile_ranges,
    double * res,
    ) nogil:
    """Poly-Exponential Curve Regression Algorithm

    This function computes and returns parameters a, b, c of the
    Poly-Exponential curve that best approximates the cloud of
    points given by coordinate vectors (x,y) and weighted by vector w (with value in [0,1]).
    a, b and c correspond to parameters in equation of the curve y = a + b * e^(-c * x).

    Parameters
    ----------
    x: np.ndarray
        Horizontal axis coordinates
    y: np.ndarray
        Vertical axis coordinates
    y: np.ndarray
        Weights for each sample
    p0: tuple[float, float, float]
        Initial guess for parameters (a, b, c)
    lower_bounds: tuple[float, float, float]
        Lower bound values for parameters (a, b, c)
    upper_bounds: tuple[float, float, float]
        Upper bound values for parameters (a, b, c)
    assume_sorted: bool
        If true, the algorithm will assume that the samples are already sorted
        along the horizontal axis, which can speed up the regression process.
    nb_quantile_ranges: int
        To speed up computations, not all samples are used. Instead they are
        reduced to core samples averaged over quantile ranges.
        This parameter allow to set the number of core samples to use.
        Higher number equals higher accuracy, but slower compute time.

    Returns
    -------
    pf: tuple[float, float, float]
        The final estimated BNPE parameters
    """


    # First reduce the number of initial samples down to a core 15 samples
    # to speed up regression
    # if not assume_sorted:
        # inds = np.argsort(x)
        # x = x[inds]
        # y = y[inds]
        # w = w[inds]

    cdef double w_sum = 0.0
    cdef int i = 0

    for i in range(nb_samples):
        w_sum += w[i]

    cdef double w_range = w_sum / nb_quantile_ranges
    cdef double cur_w_range = 0.0
    cdef double cur_w_sum = 0.0
    cdef double cur_x_sum = 0.0
    cdef double cur_y_sum = 0.0
    cdef double *xc = <double *>malloc(nb_quantile_ranges * sizeof(double))
    cdef double *yc = <double *>malloc(nb_quantile_ranges * sizeof(double))
    cdef int j = 0

    i = 0
    cdef double cur_w = w[i]
    cdef double cur_x = X[i * d]
    cdef double cur_y = X[i * d + di]
    cdef double last_w = 0.0
    while True:
        if cur_w_range + cur_w < w_range:
            cur_w_range += cur_w
            cur_w_sum += cur_w
            cur_x_sum += cur_x * cur_w
            cur_y_sum += cur_y * cur_w
            i += 1
            if i == nb_samples:
                xc[j] = cur_x_sum / cur_w_sum
                yc[j] = cur_y_sum / cur_w_sum
                break
            cur_w = w[i]
            cur_x = X[i * d]
            cur_y = X[i * d + di]
        else:
            cur_w = cur_w_range + cur_w - w_range
            last_w = 1.0 - cur_w
            xc[j] = (cur_x_sum + cur_x * last_w) / (cur_w_sum + last_w)
            yc[j] = (cur_y_sum + cur_y * last_w) / (cur_w_sum + last_w)
            j += 1
            if j == nb_quantile_ranges: break
            cur_w_range = 0.0
            cur_w_sum = 0.0
            cur_x_sum = 0.0
            cur_y_sum = 0.0

    # Conduct the regresion on the core samples

    # System generated locals
    cdef double d__1
    cdef double f,
    cdef double g[1024]
    cdef long int i__
    cdef double l[1024]
    cdef long int m = 5
    cdef long int n = 3
    cdef double u[1024], p[1024], t1, t2, wa[43251]
    cdef long int nbd[1024], iwa[3072]
    cdef long int taskValue
    cdef long int *task = &taskValue
    cdef double factr = 1e7
    cdef long int csaveValue
    cdef long int *csave = &csaveValue
    cdef double dsave[29]
    cdef long int isave[44]
    cdef long int lsave[4]
    cdef double pgtol = 1e-5
    cdef long int iprint = -1
    cdef double exp_c_d = 0.0

    for i in range(3):
        if upper_bounds[i] != INFINITY and lower_bounds[i] != INFINITY:
            nbd[i] = 2
        elif upper_bounds[i] != INFINITY:
            nbd[i] = 3
        elif lower_bounds[i] != INFINITY:
            nbd[i] = 1
        else:
            nbd[i] = 0
        l[i] = lower_bounds[i]
        u[i] = upper_bounds[i]
        p[i] = p0[i]

    task[0] = START

    while True:
        setulb(
            &n, &m, p, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task,
            &iprint, csave, lsave, isave, dsave
        )
        if task[0] >= FG and task[0] <= FG_END:
            f = 0.0 # y = a + b * e^(-c x)
            g[0] = 0.0
            g[1] = 0.0
            g[2] = 0.0
            for i in range(nb_quantile_ranges):
                exp_c_d = exp(-p[2] * xc[i])

                d__1 = yc[i] - (p[0] + p[1] * exp_c_d)
                f += d__1 * d__1

                g[0] += 2 * (p[0] + p[1] * exp_c_d - yc[i])
                g[1] += 2 * (p[0] - yc[i]) * exp_c_d + 2 * p[1] * (exp_c_d * exp_c_d)
                g[2] += -2 * p[1] * xc[i] * (exp_c_d * exp_c_d) * (p[1] + (p[0] - yc[i]) / exp_c_d)

        elif task[0] == NEW_X:
            pass
        else:
            break

    res[0] = p[0]
    res[1] = p[1]
    res[2] = p[2]
    # printf("\nRES: %f %f %f\n", p[0], p[1], p[2])
    return 0
