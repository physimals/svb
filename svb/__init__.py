"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""

from svb._version import __version__, __timestamp__
from svb.svb import SvbFit
import svb.dist as dist
import svb.model as model

__all__ = ["__version__", "__timestamp__", "SvbFit", "dist", "model"]
