"""
Definition of the voxelwise posterior distribution
"""
import tensorflow as tf

from svb.utils import LogBase

class Posterior(LogBase):
    """
    Posterior distribution
    """
    def __init__(self, **kwargs):
        """
        :param mean: Tensor of shape [V] containing the mean at each voxel
        :param var: Tensor of shape [V] containing the variance at each voxel
        """
        LogBase.__init__(self, **kwargs)

    def sample(self, nsamples):
        """
        :return: A tensor of shape [V, P, N] where V is the number
                 of voxels, P is the number of parameters in the distribution
                 (possibly 1) and N is the number of samples
        """
        raise NotImplementedError()

    def entropy(self):
        """
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
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "NormPost")
        self.mean = self.log_tf(tf.Variable(mean, dtype=tf.float32, validate_shape=False,
                                            name="%s_mean" % self.name))
        self.log_var = tf.Variable(tf.log(var), dtype=tf.float32, validate_shape=False,
                                   name="%s_log_var" % self.name)
        self.var = self.log_tf(tf.exp(self.log_var, name="%s_var" % self.name))
        self.std = self.log_tf(tf.sqrt(self.var, name="%s_std" % self.name))

    def sample(self, nsamples):
        eps = tf.random_normal((self.nvoxels, 1, nsamples), 0, 1, dtype=tf.float32)
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, 1, 1]), [1, 1, nsamples])
        sample = self.log_tf(tf.add(tiled_mean, tf.multiply(tf.reshape(self.std, [self.nvoxels, 1, 1]), eps),
                                    name="%s_sample" % self.name))
        return sample

    def entropy(self):
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
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "FactPost")

        means = [post.mean for post in self.posts]
        variances = [post.var for post in self.posts]
        self.mean = self.log_tf(tf.stack(means, axis=-1, name="%s_mean" % self.name))
        self.var = self.log_tf(tf.stack(variances, axis=-1, name="%s_var" % self.name))
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.nvoxels = posts[0].nvoxels

        # Covariance matrix is diagonal
        self.cov = tf.matrix_diag(self.var, name='%s_cov' % self.name)

    def sample(self, nsamples):
        samples = [post.sample(nsamples) for post in self.posts]
        sample = tf.concat(samples, axis=1, name="%s_sample" % self.name)
        return self.log_tf(sample)

    def entropy(self):
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

    def latent_loss(self, prior):
        """
        Analytic expression for latent loss which can be used when posterior and prior are
        Gaussian

        :param prior: Voxelwise Prior instance which defines the ``mean`` and ``cov`` voxelwise
                      attributes
        """
        prior_cov_inv = tf.matrix_inverse(prior.cov)
        mean_diff = tf.subtract(self.mean, prior.mean)

        t1 = tf.trace(tf.matmul(prior_cov_inv, self.cov))
        t2 = tf.matmul(tf.reshape(mean_diff, (self.nvoxels, 1, -1)), prior_cov_inv)
        t3 = tf.reshape(tf.matmul(t2, tf.reshape(mean_diff, (self.nvoxels, -1, 1))), [self.nvoxels])
        t4 = tf.log(tf.matrix_determinant(prior.cov, name='%s_log_det_cov' % prior.name))
        t5 = tf.log(tf.matrix_determinant(self.cov + self.cov_reg, name='%s_log_det_cov' % self.name))

        return self.log_tf(tf.identity(0.5*(t1 + t3 - self.nparams + t4 - t5), name="%s_latent_loss" % self.name))

class MVNPosterior(FactorisedPosterior):
    """
    Multivariate Normal posterior distribution
    """

    def __init__(self, posts, **kwargs):
        FactorisedPosterior.__init__(self, posts, **kwargs)

        # Covariance matrix is formed from the Cholesky decomposition
        # NB we have to create PxP variables but since we extract only the
        # off diagonal elements below we are only training on the lower
        # diagonal (excluding the diagonal itself which is comes from
        # the parameter variances provided)
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

        # Regularisation to make sure cov is invertible
        self.cov_reg = 1e-6*tf.eye(self.nparams)

        self.cov_chol = self.log_tf(self.cov_chol)
        self.cov = self.log_tf(self.cov)

    def sample(self, nsamples):
        # Use the 'reparameterization trick' to return the samples
        eps = tf.random_normal((self.nvoxels, self.nparams, nsamples), 0, 1, dtype=tf.float32)

        # NB self.cov_chol is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev.
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, self.nparams, 1]),
                             [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.matmul(self.cov_chol, eps), name="%s_sample" % self.name)
        return self.log_tf(sample)

    def entropy(self):
        det_covar = tf.matrix_determinant(self.cov + self.cov_reg, name="%s_det" % self.name) # [V]
        entropy = tf.identity(-0.5 * tf.log(det_covar), name="%s_entropy" % self.name)
        return self.log_tf(entropy)

    def state(self):
        return list(FactorisedPosterior.state(self)) + [self.off_diag_vars]

    def set_state(self, state):
        ops = list(FactorisedPosterior.set_state(self, state[:-1]))
        ops += [tf.assign(self.off_diag_vars, state[-1], validate_shape=False)]
        return ops
