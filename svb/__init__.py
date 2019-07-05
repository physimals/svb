"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""

from svb._version import __version__, __timestamp__
from svb.svb import SvbFit

__all__ = ["__version__", "__timestamp__", "SvbFit"]
