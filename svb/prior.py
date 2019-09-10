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
        :param samples: A tensor of shape [V, P, S] where V is the number
                        of voxels, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples

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

class SpatialPriorMRF(NormalPrior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring voxels using the Markov Random Field method

    This is equivalent to the Fabber 'M' type spatial prior
    """

    def __init__(self, mean, var, idx, post, nn, nn2, **kwargs):
        """
        :param mean: Tensor of shape [V] containing the prior mean at each voxel
        :param var: Tensor of shape [V] containing the prior variance at each voxel
        :param post: Posterior instance
        :param nn: Sparse tensor of shape [V, V] containing nearest neighbour lists
        :param nn2: Sparse tensor of shape [V, V] containing second nearest neighbour lists
        """
        NormalPrior.__init__(self)
        self.idx = idx

        # Save the original voxelwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour
        # voxels
        self.fixed_mean = self.mean
        self.fixed_var = self.var

        # nn and nn2 are sparse tensors of shape [V, V]. If nn[V, W] = 1 then W is
        # a nearest neighbour of W, and similarly for nn2 and second nearest neighbours
        self.nn = nn
        self.nn2 = nn2

        # Set up spatial smoothing parameter calculation from posterior and neighbour
        # lists
        self._setup_ak(post, nn, nn2)

        # Set up prior mean/variance
        self._setup_mean_var(post, nn, nn2)

    def _setup_ak(self, post, nn, nn2):
        sigmaK = tf.matrix_diag(post.cov)[:, self.idx] # [V]
        wK = post.mean[:, self.idx] # [V]
        num_nn = tf.reduce_sum(self.nn, axis=1) # [V]

        # Sum over voxels of parameter variance multiplied by number of 
        # nearest neighbours for each voxel
        trace_term = tf.reduce_sum(param_var * self.num_nn) # [1]

        # Sum of nearest neighbour mean values
        nn_means = tf.matrix_mul(self.nn, wK) # [V]
        
        # Sum over voxels 
        mean_diff 

        swk = tf.reduce_sum(wK - [:, self.idx])

        term2 = tf.reduce_sum(swk * post_mean)

    def _setup_mean_var(self, post, nn, nn2):
        pass 

class FactorisedPrior(Prior):
    """
    Prior for a collection of parameters where there is no prior covariance

    In this case the mean log PDF can be summed from the contributions of each
    parameter
    """

    def __init__(self, priors, **kwargs):
        Prior.__init__(self)
        self.priors = priors
        self.name = kwargs.get("name", "FactPrior")
        self.nparams = len(priors)

        means = [prior.mean for prior in self.priors]
        variances = [prior.var for prior in self.priors]
        self.mean = self.log_tf(tf.stack(means, axis=-1, name="%s_mean" % self.name))
        self.var = self.log_tf(tf.stack(variances, axis=-1, name="%s_var" % self.name))
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.nvoxels = priors[0].nvoxels

        # Define a diagonal covariance matrix for convenience
        self.cov = tf.matrix_diag(self.var, name='%s_cov' % self.name)

    def mean_log_pdf(self, samples):
        nvoxels = tf.shape(samples)[0]

        mean_log_pdf = tf.zeros([nvoxels], dtype=tf.float32)
        for idx, prior in enumerate(self.priors):
            param_samples = tf.slice(samples, [0, idx, 0], [-1, 1, -1])
            param_logpdf = prior.mean_log_pdf(param_samples)
            mean_log_pdf = tf.add(mean_log_pdf, param_logpdf)
        return mean_log_pdf
    