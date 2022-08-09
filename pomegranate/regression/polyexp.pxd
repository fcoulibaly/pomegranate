
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
    ) nogil;
