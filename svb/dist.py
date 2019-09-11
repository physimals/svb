"""
Distributions that can be applied to a model parameter
"""
import math

import tensorflow as tf

from .prior import NormalPrior, SpatialPriorMRF
from .posterior import NormalPosterior
from .utils import LogBase

class Identity:
    """
    Base class for variable transformations which defines just a
    simple identity transformation
    """

    def int_values(self, ext_values):
        """
        Convert internal (inferred) values to external
        (model-visible) values
        """
        return ext_values

    def int_moments(self, ext_mean, ext_var):
        """
        Convert internal (inferred) mean/variance to external
        (model-visible) mean/variance
        """
        return ext_mean, ext_var

    def ext_values(self, int_values):
        """
        Convert external (model) values to internal
        (inferred) values
        """
        return int_values

    def ext_moments(self, int_mean, int_var):
        """
        Convert the external (model) mean/variance to internal
        (inferred) mean/variance
        """
        return int_mean, int_var

class Log(Identity):
    """
    Log-transform used for log-normal distribution
    """
    def __init__(self, geom=True):
        self._geom = geom

    def int_values(self, ext_values):
        return tf.log(ext_values)

    def int_moments(self, ext_mean, ext_var):
        if self._geom:
            return math.log(ext_mean), math.log(ext_var)
        else:
            # See https://uk.mathworks.com/help/stats/lognstat.html
            return math.log(ext_mean**2/math.sqrt(ext_var + ext_mean**2)), math.log(ext_var/ext_mean**2 + 1)

    def ext_values(self, int_values):
        return tf.exp(int_values)

    def ext_moments(self, int_mean, int_var):
        if self._geom:
            return math.exp(int_mean), math.exp(int_var)
        else:
            raise NotImplementedError()

class Abs(Identity):
    """
    Absolute value transform used for folded normal distribution
    """
    def ext_values(self, int_values):
        return tf.abs(int_values)

    def ext_moments(self, int_mean, int_var):
        raise NotImplementedError()

class Dist(LogBase):
    """
    A parameter distribution
    """
    def voxelwise_prior(self, nvoxels, **kwargs):
        """
        :return: Prior instance for the given number of voxels
        """
        raise NotImplementedError()

    def voxelwise_posterior(self, param, t, data, initialise=None, **kwargs):
        """
        :return: Posterior instance for the given number of voxels
        """
        raise NotImplementedError()

class Normal(Dist):
    """
    Gaussian-based distribution

    The distribution of a parameter has an *underlying* Gaussian
    distribution but may apply a transformation on top of this
    to form the *model* distribution.

    We force subclasses to implement the required methods rather
    than providing a default implementation
    """
    def __init__(self, ext_mean, ext_var, transform=Identity()):
        """
        Constructor.

        Sets the distribution mean, variance and std.dev.

        Note that these are the mean/variance of the *model*
        distribution, not the underlying Gaussian - the latter are
        returned by the ``int_mean`` and ``int_var`` methods
        """
        Dist.__init__(self)
        self.transform = transform
        self.mean, self.var = self.transform.int_moments(ext_mean, ext_var)
        self.sd = math.sqrt(self.var)

    def voxelwise_prior(self, nvoxels, **kwargs):
        mean = tf.fill([nvoxels], self.mean)
        var = tf.fill([nvoxels], self.var)
        self.log.info("Parameter %s: Normal prior (%f, %f)", kwargs.get("name", "unknown"), self.mean, self.var)
        return NormalPrior(mean, var, **kwargs)

    def voxelwise_posterior(self, param, t, data, initialise=None, **kwargs):
        nvoxels = tf.shape(data)[0]
        initial_mean, initial_var = None, None
        if initialise is not None:
            initial_mean, initial_var = initialise(param, t, data)

        if initial_mean is None:
            initial_mean = tf.fill([nvoxels], self.mean)
            self.log.info("Parameter %s: Initial posterior mean %f", kwargs.get("name", "unknown"), self.mean)
        else:
            initial_mean = self.transform.int_values(initial_mean)
            self.log.info("Parameter %s: Voxelwise initial posterior mean", kwargs.get("name", "unknown"))

        if initial_var is None:
            initial_var = tf.fill([nvoxels], self.var)
            self.log.info("Parameter %s: Initial posterior variance %f", kwargs.get("name", "unknown"), self.mean)
        else:
            initial_var = self.transform.int_values(initial_var)
            self.log.info("Parameter %s: Voxelwise initial posterior variance", kwargs.get("name", "unknown"))

        return NormalPosterior(initial_mean, initial_var, **kwargs)

class LogNormal(Normal):
    """
    Log of the parameter is distributed as a Gaussian.

    This is one means of ensuring that a parameter is always > 0.
    """

    def __init__(self, mean, var, geom=True, **kwargs):
        Normal.__init__(self, mean, var, transform=Log(geom), **kwargs)

class FoldedNormal(Normal):
    """
    Distribution where the probability density
    is zero for negative values and the sum of Gaussian
    densities for the value and its negative otherwise

    This is a fancy way of saying we take the absolute
    value of the underlying distribution as the model
    distribution value.
    """

    def __init__(self, mean, var, **kwargs):
        Normal.__init__(self, mean, var, transform=Abs(), **kwargs)
