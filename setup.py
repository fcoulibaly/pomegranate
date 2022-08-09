from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
# https://stackoverflow.com/a/11181607/541202
# import __builtin__ as __builtins__

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    ext = 'c'
else:
    use_cython = True
    ext = 'pyx'

filenames = [ 
    "base",
    "bayes",
    "BayesianNetwork",
    "MarkovNetwork",
    "FactorGraph",
    "hmm",
    "gmm",
    "kmeans",
    "NaiveBayes",
    "BayesClassifier",
    "MarkovChain",
    "utils",
    "parallel"
]

regression_filenames = [
    "lbfgsb.c",
    "linesearch.c",
    "linpack.c",
    "miniCBLAS.c",
    "print.c",
    "subalgorithms.c",
    "timer.c",
    "polyexp.pyx",
    "polyexp.pxd",
]

distributions = [
    'distributions',
    'UniformDistribution',
    'BernoulliDistribution',
    'NormalDistribution',
    'LogNormalDistribution',
    'ExponentialDistribution',
    'BetaDistribution',
    'TrueBetaDistribution',
    'GammaDistribution',
    'DiscreteDistribution',
    'PoissonDistribution',
    'KernelDensities',
    'IndependentComponentsDistribution',
    'MultivariateGaussianDistribution',
    'DirichletDistribution',
    'ConditionalProbabilityTable',
    'JointProbabilityTable',
    'PolyExpBetaNormalDistribution',
]

if not use_cython:
    extensions = [
        Extension("pomegranate.{}".format( name ), [ "pomegranate/{}.{}".format(name, ext) ]) for name in filenames
    ] + [
        # Extension('pomegranate.distributions.PolyExpBetaNormalDistribution',
                  # [ "pomegranate/regression{}.{}".format(name, ext) for name in filenames ] +
                  # [ "pomegranate/distributions/PolyExpBetaNormalDistribution.{}".format(dist, ext) ]
                  # )
    ] + [
        Extension("pomegranate.distributions.{}".format(dist), ["pomegranate/distributions/{}.{}".format(dist, ext)]) for dist in distributions
    ] + [
        Extension("pomegranate.regression.polyexp", [ "pomegranate/regression/polyexp.pyx" ])
    ]
else:
    extensions = [
            Extension("pomegranate.*", ["pomegranate/*.pyx"]),
            # Extension('pomegranate.distributions.PolyExpBetaNormalDistribution',
                      # [ "pomegranate/regression/{}".format(name) for name in regression_filenames ] +
                      # [ "pomegranate/distributions/PolyExpBetaNormalDistribution.pyx"]
                      # ),
    # Extension("pomegranate.distributions.*", ["pomegranate/distributions/*.pyx"])
    ] + [
        Extension("pomegranate.distributions.{}".format(dist), ["pomegranate/distributions/{}.{}".format(dist, ext)]) for dist in distributions
    ] + [
        Extension("pomegranate.regression.polyexp", [ "pomegranate/regression/{}".format(name) for name in regression_filenames ]
        )
    ]

    extensions = cythonize(extensions, compiler_directives={'language_level' : "2"})

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        if hasattr(__builtins__, '__NUMPY_SETUP__'):
            __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='pomegranate',
    version='0.14.4',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=[
        'pomegranate',
        'pomegranate/distributions',
    ],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='MIT',
    description='Pomegranate is a graphical models library for Python, implemented in Cython for speed.',
    ext_modules=extensions,
    cmdclass={'build_ext':build_ext},
    setup_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.20.0",
        "scipy >= 0.17.0"
    ],
    install_requires=[
        "numpy >= 1.20.0",
        "joblib >= 0.9.0b4",
        "networkx >= 2.0",
        "scipy >= 0.17.0",
        "pyyaml"
    ],
    extras_require={
        "Plotting": ["pygraphviz", "matplotlib"],
        "GPU": ["cupy"],
    },
    test_suite = 'nose.collector',
    package_data={
        'pomegranate': ['*.pyd', '*.pxd'],
        'pomegranate/distributions': ['*.pyd', '*.pxd'],
    },
    include_package_data=True,
    zip_safe=False,
)
