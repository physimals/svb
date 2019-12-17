"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""

from ._version import __version__, __timestamp__
from .svb import SvbFit
from .data import DataModel
from .model import Model
from .models import get_model_class

__all__ = [
    "__version__",
    "__timestamp__",
    "SvbFit",
    "DataModel",
    "Model",
    "get_model_class",
]
