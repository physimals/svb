"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""

from ._version import __version__, __timestamp__
from .svb import SvbFit

__all__ = [
    "__version__",
    "__timestamp__",
    "SvbFit",
]
