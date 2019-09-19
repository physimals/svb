"""
Definition of prior distribution
"""
import tensorflow as tf

from .utils import LogBase
from .dist import Normal

PRIOR_TYPE_NONSPATIAL = "N"
PRIOR_TYPE_SPATIAL_MRF = "M"

def get_voxelwise_prior(param, nvoxels, **kwargs):
    """
    Factory method to return a voxelwise prior
    """
    prior = None
    if isinstance(param.prior_dist, Normal):
        if param.priortype == "N":
            prior = NormalPrior(nvoxels, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.priortype == "M":
            prior = MRFSpatialPrior(nvoxels, param.prior_dist.mean, param.prior_dist.var, **kwargs)

    if prior is not None:
        return prior
    else:
        raise ValueError("Can't create prior type %s for distribution %s - unrecognized combination" % (param.priortype, param.prior_dist))

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

    def __init__(self, nvoxels, mean, var, **kwargs):
        """
        :param mean: Prior mean value
        :param var: Prior variance
        """
        Prior.__init__(self)
        self.name = kwargs.get("name", "NormalPrior")
        self.nvoxels = nvoxels
        self.scalar_mean = mean
        self.scalar_var = var
        self.mean = tf.fill([nvoxels], mean, name="%s_mean" % self.name)
        self.var = tf.fill([nvoxels], var, name="%s_var" % self.name)

    def mean_log_pdf(self, samples):
        dx = tf.subtract(samples, tf.reshape(self.mean, [self.nvoxels, 1, 1])) # [V, 1, N]
        z = tf.div(tf.square(dx), tf.reshape(self.var, [self.nvoxels, 1, 1])) # [V, 1, N]
        log_pdf = -0.5*z # [V, 1, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nvoxels]) # [V]
        return mean_log_pdf

    def __str__(self):
        return "Normal prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

class MRFSpatialPrior(NormalPrior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring voxels using the Markov Random Field method

    This is equivalent to the Fabber 'M' type spatial prior
    """

    def __init__(self, nvoxels, mean, var, idx=None, post=None, nn=None, n2=None, **kwargs):
        """
        :param mean: Tensor of shape [V] containing the prior mean at each voxel
        :param var: Tensor of shape [V] containing the prior variance at each voxel
        :param post: Posterior instance
        :param nn: Sparse tensor of shape [V, V] containing nearest neighbour lists
        :param n2: Sparse tensor of shape [V, V] containing second nearest neighbour lists
        """
        NormalPrior.__init__(self, nvoxels, mean, var, name="MRFSpatialPrior")
        self.idx = idx

        # Save the original voxelwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour voxels
        self.fixed_mean = self.mean
        self.fixed_var = self.var

        # nn and n2 are sparse tensors of shape [V, V]. If nn[V, W] = 1 then W is
        # a nearest neighbour of W, and similarly for n2 and second nearest neighbours
        self.nn = nn
        self.n2 = n2

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        self._setup_ak(post, nn, n2)

        # Set up prior mean/variance
        self._setup_mean_var(post, nn, n2)

    def __str__(self):
        return "Spatial MRF prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

    def _setup_ak(self, post, nn, n2):
        # This is the equivalent of CalculateAk in Fabber
        #
        # Some of this could probably be better done using linalg
        # operations but bear in mind this is one parameter only

        self.sigmaK = self.log_tf(tf.matrix_diag_part(post.cov)[:, self.idx], name="sigmak") # [V]
        self.wK = self.log_tf(post.mean[:, self.idx], name="wk") # [V]
        self.num_nn = self.log_tf(tf.sparse_reduce_sum(self.nn, axis=1), name="num_nn") # [V]

        # Sum over voxels of parameter variance multiplied by number of 
        # nearest neighbours for each voxel
        trace_term = self.log_tf(tf.reduce_sum(self.sigmaK * self.num_nn), name="trace") # [1]

        # Sum of nearest and next-nearest neighbour mean values
        self.sum_means_nn = self.log_tf(tf.reshape(tf.sparse_tensor_dense_matmul(self.nn, tf.reshape(self.wK, (-1, 1))), (-1,)), name="wksum") # [V]
        self.sum_means_n2 = self.log_tf(tf.reshape(tf.sparse_tensor_dense_matmul(self.n2, tf.reshape(self.wK, (-1, 1))), (-1,)), name="contrib8") # [V]
        
        # Voxel parameter mean multipled by number of nearest neighbours
        wknn = self.log_tf(self.wK * self.num_nn, name="wknn") # [V]

        swk = self.log_tf(wknn - self.sum_means_nn, name="swk") # [V]

        term2 = self.log_tf(tf.reduce_sum(swk * self.wK), name="term2") # [1]

        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hk = tf.multiply(tf.to_float(self.nvoxels), 0.5) + 1.0
        self.ak = self.log_tf(gk * hk, name="%s_ak" % self.name, force=True)

    def _setup_mean_var(self, post, nn, n2):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = self.log_tf(8*self.sum_means_nn, name="contrib_nn") # [V]
        contrib_n2 = self.log_tf(-self.sum_means_n2, name="contrib_n2") # [V]
        
        spatial_mean = self.log_tf(contrib_nn / (8*self.num_nn), name="spatial_mean")
        spatial_prec = self.log_tf(self.num_nn * self.ak, name="spatial_prec")

        self.var = self.log_tf(1 / (1/self.fixed_var + spatial_prec), name="%s_var" % self.name)
        #self.var = self.fixed_var
        self.mean = self.log_tf(self.var * spatial_prec * spatial_mean, name="%s_mean" % self.name)
        #self.mean = self.fixed_mean + self.ak

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
    
    def log_det_cov(self):
        return tf.log(tf.matrix_determinant(self.cov), name='%s_log_det_cov' % self.name)
