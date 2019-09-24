"""
Definition of the voxelwise posterior distribution
"""
import tensorflow as tf

from .utils import LogBase
from . import dist

def get_voxelwise_posterior(param, t, data, **kwargs):
    """
    Factory method to return a voxelwise posterior

    :param param: svb.parameter.Parameter instance
    :
    """
    nvoxels = tf.shape(data)[0]
    initial_mean, initial_var = None, None
    if param.post_initialise is not None:
        initial_mean, initial_var = param.post_initialise(param, t, data)

    if initial_mean is None:
        initial_mean = tf.fill([nvoxels], param.post_dist.mean)
        #self.log.info("Parameter %s: Initial posterior mean %f", param.name, param.post_dist.mean)
    else:
        initial_mean = param.post_dist.transform.int_values(initial_mean)
        #self.log.info("Parameter %s: Voxelwise initial posterior mean", param.name)

    if initial_var is None:
        initial_var = tf.fill([nvoxels], param.post_dist.var)
        #self.log.info("Parameter %s: Initial posterior variance %f", param.name, param.post_dist.mean)
    else:
        initial_var = param.post_dist.transform.int_values(initial_var)
        #self.log.info("Parameter %s: Voxelwise initial posterior variance", param.name)

    if isinstance(param.post_dist, dist.Normal):
        return NormalPosterior(initial_mean, initial_var, name=param.name, **kwargs)
    else:
        raise ValueError("Can't create posterior for distribution: %s" % param.post_dist)
        
class Posterior(LogBase):
    """
    Posterior distribution
    """

    def sample(self, nsamples):
        """
        :param nsamples: Number of samples to return per voxel / parameter

        :return: A tensor of shape [V, P, S] where V is the number
                 of voxels, P is the number of parameters in the distribution
                 (possibly 1) and S is the number of samples
        """
        raise NotImplementedError()

    def entropy(self, samples=None):
        """
        :param samples: A tensor of shape [V, P, S] where V is the number
                        of voxels, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples.
                        This parameter may or may not be used in the calculation.
                        If it is required, the implementation class must check
                        that it is provided

        :return Tensor of shape [V] containing voxelwise distribution entropy
        """
        raise NotImplementedError()

    def state(self):
        """
        :return Sequence of tf.Tensor objects containing the state of all variables in this
                posterior. The tensors returned will be evaluated to create a savable state
                which may then be passed back into set_state()
        """
        raise NotImplementedError()

    def set_state(self, state):
        """
        :param state: State of variables in this posterior, as returned by previous call to state()

        :return Sequence of tf.Operation objects containing which will set the variables in
                this posterior to the specified state
        """
        raise NotImplementedError()

    def log_det_cov(self):
        raise NotImplementedError()

class NormalPosterior(Posterior):
    """
    Posterior distribution for a single voxelwise parameter with a normal
    distribution
    """

    def __init__(self, mean, var, **kwargs):
        """
        :param mean: Tensor of shape [V] containing the mean at each voxel
        :param var: Tensor of shape [V] containing the variance at each voxel
        """
        Posterior.__init__(self, **kwargs)
        self.nvoxels = tf.shape(mean)[0]
        self.name = kwargs.get("name", "NormPost")
        self.mean = self.log_tf(tf.Variable(mean, dtype=tf.float32, validate_shape=False,
                                            name="%s_mean" % self.name))
        #self.mean = tf.where(tf.is_nan(self.mean), tf.ones_like(self.mean), self.mean)
        self.log_var = tf.Variable(tf.log(tf.cast(var, dtype=tf.float32)), validate_shape=False,
                                   name="%s_log_var" % self.name)
        self.var = self.log_tf(tf.exp(self.log_var, name="%s_var" % self.name))
        #self.var = tf.where(tf.is_nan(self.var), tf.ones_like(self.var), self.var)
        self.std = self.log_tf(tf.sqrt(self.var, name="%s_std" % self.name))

    def sample(self, nsamples):
        eps = tf.random_normal((self.nvoxels, 1, nsamples), 0, 1, dtype=tf.float32)
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, 1, 1]), [1, 1, nsamples])
        sample = self.log_tf(tf.add(tiled_mean, tf.multiply(tf.reshape(self.std, [self.nvoxels, 1, 1]), eps),
                                    name="%s_sample" % self.name))
        return sample

    def entropy(self, _samples=None):
        entropy = tf.identity(-0.5 * tf.log(self.var), name="%s_entropy" % self.name)
        return self.log_tf(entropy)

    def state(self):
        return [self.mean, self.log_var]

    def set_state(self, state):
        return [
            tf.assign(self.mean, state[0]),
            tf.assign(self.log_var, state[1])
        ]

class FactorisedPosterior(Posterior):
    """
    Posterior distribution for a set of parameters with no covariance
    """

    def __init__(self, posts, **kwargs):
        Posterior.__init__(self, **kwargs)
        self.posts = posts
        self.nparams = len(self.posts)
        self.name = kwargs.get("name", "FactPost")

        if "init" in kwargs:
            # We have been provided with an initialization posterior. Extract the mean and diagonal of the
            # covariance and use that as the initial values of the mean and variance variable tensors
            mean, cov = kwargs["init"]
            # Check initialisation posterior has correct number of parameters and voxels
            if mean.shape[1] != len(self.posts):
                raise ValueError("Initializing posterior with %i parameters using posterior with %i parameters" % (mean.shape[1], len(self.posts)))
            #if mean.shape[0] != posts[0].nvoxels:
            #    raise ValueError("Initializing posterior with %i voxels using posterior with %i voxels" % (mean.shape[0], posts[0].nvoxels))
            mean = tf.Variable(mean, dtype=tf.float32, validate_shape=False)
            var = tf.Variable(tf.matrix_diag_part(cov), dtype=tf.float32, validate_shape=False)
        else:
            means = [post.mean for post in self.posts]
            variances = [post.var for post in self.posts]
            mean = tf.stack(means, axis=-1, name="%s_mean" % self.name)
            var = tf.stack(variances, axis=-1, name="%s_var" % self.name)

        self.mean = self.log_tf(tf.identity(mean, name="%s_mean" % self.name))
        self.var = self.log_tf(tf.identity(var, name="%s_var" % self.name))
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.nvoxels = posts[0].nvoxels

        # Covariance matrix is diagonal
        self.cov = tf.matrix_diag(self.var, name='%s_cov' % self.name)

        # Regularisation to make sure cov is invertible. Note that we do not
        # need this for a diagonal covariance matrix but it is useful for
        # the full MVN covariance which shares some of the calculations
        self.cov_reg = 1e-5*tf.eye(self.nparams)

    def sample(self, nsamples):
        samples = [post.sample(nsamples) for post in self.posts]
        sample = tf.concat(samples, axis=1, name="%s_sample" % self.name)
        return self.log_tf(sample)

    def entropy(self, _samples=None):
        entropy = tf.zeros([self.nvoxels], dtype=tf.float32)
        for post in self.posts:
            entropy = tf.add(entropy, post.entropy(), name="%s_entropy" % self.name)
        return self.log_tf(entropy)

    def state(self):
        state = []
        for post in self.posts:
            state.extend(post.state())
        return state

    def set_state(self, state):
        ops = []
        for idx, post in enumerate(self.posts):
            ops += post.set_state(state[idx*2:idx*2+2])
        return ops

    def log_det_cov(self):
        """
        Determinant of diagonal matrix is product of diagonal entries
        """
        return tf.reduce_sum(tf.log(self.var), axis=1, name='%s_log_det_cov' % self.name)
        #return tf.log(tf.matrix_determinant(self.cov), name='%s_log_det_cov' % self.name)

    def latent_loss(self, prior):
        """
        Analytic expression for latent loss which can be used when posterior and prior are
        Gaussian

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

        :param prior: Voxelwise Prior instance which defines the ``mean`` and ``cov`` voxelwise
                      attributes
        """
        prior_cov_inv = tf.matrix_inverse(prior.cov)
        mean_diff = tf.subtract(self.mean, prior.mean)

        term1 = tf.trace(tf.matmul(prior_cov_inv, self.cov))
        term2 = tf.matmul(tf.reshape(mean_diff, (self.nvoxels, 1, -1)), prior_cov_inv)
        term3 = tf.reshape(tf.matmul(term2, tf.reshape(mean_diff, (self.nvoxels, -1, 1))), [self.nvoxels])
        term4 = prior.log_det_cov()
        term5 = self.log_det_cov()

        return self.log_tf(tf.identity(0.5*(term1 + term3 - self.nparams + term4 - term5), name="%s_latent_loss" % self.name))

class MVNPosterior(FactorisedPosterior):
    """
    Multivariate Normal posterior distribution
    """

    def __init__(self, posts, **kwargs):
        FactorisedPosterior.__init__(self, posts, **kwargs)

        # The full covariance matrix is formed from the Cholesky decomposition
        # to ensure that it remains positive definite.
        #
        # To achieve this, we have to create PxP tensor variables for
        # each voxel, but we then extract only the lower triangular
        # elements and train only on these. The diagonal elements
        # are constructed by the FactorisedPosterior
        if "init" in kwargs:
            # We are initializing from an existing posterior.
            # The FactorizedPosterior will already have extracted the mean and
            # diagonal of the covariance matrix - we need the Cholesky decomposition
            # of the covariance to initialize the off-diagonal terms
            _mean, cov = kwargs["init"]
            covar_init = tf.cholesky(cov, dtype=tf.float32)
        else:
            covar_init = tf.zeros([self.nvoxels, self.nparams, self.nparams], dtype=tf.float32)

        self.off_diag_vars = self.log_tf(tf.Variable(covar_init, validate_shape=False,
                                                     name='%s_off_diag_vars' % self.name))
        self.off_diag_cov_chol = tf.matrix_set_diag(tf.matrix_band_part(self.off_diag_vars, -1, 0),
                                                    tf.zeros([self.nvoxels, self.nparams]),
                                                    name='%s_off_diag_cov_chol' % self.name)

        # Combine diagonal and off-diagonal elements into full matrix
        self.cov_chol = tf.add(tf.matrix_diag(self.std), self.off_diag_cov_chol,
                               name='%s_cov_chol' % self.name)

        # Form the covariance matrix from the chol decomposition
        self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol,
                             name='%s_cov' % self.name)

        self.cov_chol = self.log_tf(self.cov_chol)
        self.cov = self.log_tf(self.cov)

    def log_det_cov(self):
        return self.log_tf(tf.multiply(2.0, tf.reduce_sum(tf.log(tf.matrix_diag_part(self.cov_chol)), axis=1), name="%s_det_cov" % self.name), force=False)

    def sample(self, nsamples):
        # Use the 'reparameterization trick' to return the samples
        eps = tf.random_normal((self.nvoxels, self.nparams, nsamples), 0, 1, dtype=tf.float32, name="eps")

        # NB self.cov_chol is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev.
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, self.nparams, 1]),
                             [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.matmul(self.cov_chol, eps), name="%s_sample" % self.name)
        return self.log_tf(sample)

    def entropy(self, _samples=None):
        entropy = tf.identity(-0.5 * self.log_det_cov(), name="%s_entropy" % self.name)
        return self.log_tf(entropy)

    def state(self):
        return list(FactorisedPosterior.state(self)) + [self.off_diag_vars]

    def set_state(self, state):
        ops = list(FactorisedPosterior.set_state(self, state[:-1]))
        ops += [tf.assign(self.off_diag_vars, state[-1], validate_shape=False)]
        return ops
