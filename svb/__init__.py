"""
Implementation of Stochastic Variational Bayesian inference for fitting timeseries data
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .svb import SvbFit
from .data import DataModel
from .model import Model, get_model_class

__all__ = [   "__version__",
    "__timestamp__",
    "SvbFit",
    "DataModel",
    "Model",
    "get_model_class",
]
