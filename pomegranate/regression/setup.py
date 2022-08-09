from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension('wrapper', ['lbfgsb.c', 'miniCBLAS.c', 'linesearch.c', 'linpack.c', 'print.c', 'subalgorithms.c', 'timer.c', 'wrapper.pyx'])]
setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
)
