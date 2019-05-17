"""
Definition of prior distribution
"""
import tensorflow as tf

from svb.utils import LogBase

class Prior(LogBase):
    """
    Base class for a prior, defining methods that must be implemented
    """

    def mean_log_pdf(self, samples):
        """
        :param samples: A tensor of shape [V, P, N] where V is the number
                        of voxels, P is the number of parameters in the prior
                        (possibly 1) and N is the number of samples

        :return: A tensor of shape [V] where V is the number of voxels
                 containing the mean log PDF of the parameter samples
                 provided
        """
        raise NotImplementedError()

class NormalPrior(Prior):
    """
    Prior based on a voxelwise univariate normal distribution
    """

    def __init__(self, mean, var, **kwargs):
        """
        :param mean: Tensor of shape [V] containing the prior mean at each voxel
        :param var: Tensor of shape [V] containing the prior variance at each voxel
        """
        Prior.__init__(self)
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "NormalPrior")
        self.nvoxels = tf.shape(mean)[0]
        self.mean = tf.identity(mean, name="%s_mean" % self.name)
        self.var = tf.identity(var, name="%s_var" % self.name)

    def mean_log_pdf(self, samples):
        dx = tf.subtract(samples, tf.reshape(self.mean, [self.nvoxels, 1, 1])) # [V, 1, N]
        z = tf.div(tf.square(dx), tf.reshape(self.var, [self.nvoxels, 1, 1])) # [V, 1, N]
        log_pdf = -0.5*z # [V, 1, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nvoxels]) # [V]
        return mean_log_pdf

class FactorisedPrior(Prior):
    """
    Prior for a collection of parameters where there is no prior covariance

    In this case the mean log PDF can be summed from the contributions of each
    parameter
    """

    def __init__(self, priors, **kwargs):
        self.priors = priors
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "FactPrior")
        self.nparams = len(priors)

    def mean_log_pdf(self, samples):
        nvoxels = tf.shape(samples)[0]

        mean_log_pdf = tf.zeros([nvoxels], dtype=tf.float32)
        for idx, prior in enumerate(self.priors):
            param_samples = tf.slice(samples, [0, idx, 0], [-1, 1, -1])
            #param_samples = tf.Print(param_samples, [tf.shape(param_samples), param_samples], "param_samples")
            param_logpdf = prior.mean_log_pdf(param_samples)
            #param_logpdf = tf.Print(param_logpdf, [param_logpdf], "param_logpdf")
            mean_log_pdf = tf.add(mean_log_pdf, param_logpdf)
        return mean_log_pdf
    