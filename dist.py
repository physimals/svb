"""
Distributions that can be applied to a model parameter
"""
import math

import tensorflow as tf

class Dist:
    """
    A parameter distribution

    The distribution of a parameter has an *underlying* Gaussian
    distribution but may apply a transformation on top of this
    to form the *model* distribution.

    We force subclasses to implement the required methods rather
    than providing a default implementation
    """

    def __init__(self, mean, var=None, sd=None):
        """
        Constructor. 
        
        Sets the distribution mean, variance and std.dev.
        Either variance or SD may be given - the other is derived.

        Note that these are the mean/variance of the *model*
        distribution, not the underlying Gaussian - the latter are
        returned by the ``nmean`` and ``nvar`` methods
        """
        self.mean = mean
        if var is not None:
            self.var = var
            self.sd = math.sqrt(self.var)
        elif sd is not None:
            self.var = sd ** 2
            self.sd = sd
        else:
            raise ValueError("Neither variance nor std. dev. were given")
    
    @property
    def nmean(self):
        """
        :return: The mean of the underlying Gaussian distribution
        """
        raise NotImplementedError("nmean")

    @property
    def nvar(self):
        """
        :return: The variance of the underlying Gaussian distribution
        """
        raise NotImplementedError("nvar")

    def tomodel(self, values):
        """
        Convert values from the underlying Gaussian distribution to the
        model distribution.

        :param values: tf.Tensor. All of the values in the tensor are
                       interpreted as values from the underlying distribution
        :return: tf.Tensor of same shape as ``values`` containing transformed
                 values
        """
        raise NotImplementedError("tomodel")

    def tonormal(self, values):
        """
        Convert values from the model distribution to the underlying Gaussian
        distribution.
        
        This is the inverse of ``tomodel``. Currently we are not using
        this - is it needed at all?

        :param values: tf.Tensor. All of the values in the tensor are
                       interpreted as values from the distribution
        :return: tf.Tensor of same shape as ``values`` containing 
                 corresponding values from the underlying Gaussian distribution
        """
        raise NotImplementedError("tonormal")

class Normal(Dist):
    """
    Parameter distribution is a Gaussian

    This means the 'transformation' methods are just identities
    """

    def __init__(self, *args, **kwargs):
        Dist.__init__(self, *args, **kwargs)

    @property
    def nmean(self):
        return self.mean

    @property
    def nvar(self):
        return self.var

    def tomodel(self, values):
        return values

    def tonormal(self, values):
        return values

class LogNormal(Dist):
    """
    Log of the parameter is distributed as a Gaussian.

    This is one means of ensuring that a parameter is always > 0.
    """

    def __init__(self, *args, **kwargs):
        Dist.__init__(self, *args, **kwargs)

    @property
    def nmean(self):
        # See https://uk.mathworks.com/help/stats/lognstat.html
        return math.log(self.mean**2/math.sqrt(self.var + self.mean**2))

    @property
    def nvar(self):
        # See https://uk.mathworks.com/help/stats/lognstat.html
        return math.log(self.var/self.mean**2 + 1)

    def tomodel(self, values):
        return tf.exp(values)

    def tonormal(self, values):
        return tf.log(values)

class FoldedNormal(Dist):
    """
    Distribution where the probability density
    is zero for negative values and the sum of Gaussian
    densities for the value and its negative otherwise

    This is a fancy way of saying we take the absolute
    value of the underlying distribution as the model
    distribution value.
    """

    def __init__(self, *args, **kwargs):
        Dist.__init__(self, *args, **kwargs)

    @property
    def nmean(self):
        """ FIXME not as simple as this"""
        return self.mean

    @property
    def nvar(self):
        """ FIXME not as simple as this"""
        return self.var

    def tomodel(self, values):
        return tf.abs(values)

    def tonormal(self, values):
        """
        Since distribution values are always positive we will
        not change them. FIXME this seems wrong but does it
        matter for this application?
        """
        return values
