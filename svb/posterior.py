"""
Definition of the voxelwise posterior distribution
"""
import tensorflow as tf

from .utils import debug

class Posterior:
    """
    Posterior distribution
    """

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
        self.nvoxels = tf.shape(mean)[0]
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "NormPost")
        self.mean = debug(self, tf.Variable(mean, dtype=tf.float32, validate_shape=False, name="%s_mean" % self.name))
        self.var = debug(self, tf.exp(tf.Variable(tf.log(var), dtype=tf.float32, validate_shape=False), name="%s_var" % self.name))
        self.std = debug(self, tf.sqrt(self.var, name="%s_std" % self.name))

    def sample(self, nsamples):
        eps = tf.random_normal((self.nvoxels, 1, nsamples), 0, 1, dtype=tf.float32)
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, 1, 1]), [1, 1, nsamples])
        sample = debug(self, tf.add(tiled_mean, tf.multiply(tf.reshape(self.std, [self.nvoxels, 1, 1]), eps),
                        name="%s_sample" % self.name))
        return sample

    def entropy(self):
        entropy = tf.identity(-0.5 * tf.log(self.var), name="%s_entropy" % self.name)
        return debug(self, entropy)

class FactorisedPosterior(Posterior):
    """
    Posterior distribution for a set of parameters with no covariance
    """

    def __init__(self, posts, **kwargs):
        self.posts = posts
        self.nparams = len(self.posts)
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "FactPost")

        means = [post.mean for post in self.posts]
        variances = [post.var for post in self.posts]
        self.mean = debug(self, tf.stack(means, axis=-1, name="%s_mean" % self.name))
        self.var = debug(self, tf.stack(variances, axis=-1, name="%s_var" % self.name))
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.nvoxels = tf.shape(self.mean)[0]

        # Covariance matrix is diagonal
        self.cov = tf.matrix_diag(self.var, name='%s_cov' % self.name)

    def sample(self, nsamples):
        samples = [post.sample(nsamples) for post in self.posts]
        sample = tf.concat(samples, axis=1, name="%s_sample" % self.name)
        return debug(self, sample)

    def entropy(self):
        entropy = tf.zeros([self.nvoxels], dtype=tf.float32)
        for post in self.posts:
            entropy = tf.add(entropy, post.entropy(), name="%s_entropy" % self.name)
        return debug(self, entropy)

class MVNPosterior(Posterior):
    """
    Multivariate Normal posterior distribution
    """

    def __init__(self, posts, **kwargs):
        self.posts = posts
        self.nparams = len(self.posts)
        self.debug = kwargs.get("debug", False)
        self.name = kwargs.get("name", "FactPost")

        means = [post.mean for post in self.posts]
        variances = [post.var for post in self.posts]
        self.mean = debug(self, tf.stack(means, axis=-1, name="%s_mean" % self.name))
        self.var = debug(self, tf.stack(variances, axis=-1, name="%s_var" % self.name))
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.nvoxels = tf.shape(self.mean)[0]

        # Covariance matrix is formed from the Cholesky decomposition
        # NB we have to create PxP variables but since we extract only the
        # off diagonal elements below we are only training on the lower
        # diagonal (excluding the diagonal itself which is comes from
        # the parameter variances provided)
        covar_init = tf.zeros([self.nvoxels, self.nparams, self.nparams])
        self.off_diag_vars = tf.Variable(covar_init, validate_shape=False,
                                         name='%s_off_diag_vars' % self.name)
        self.off_diag_cov_chol = tf.matrix_set_diag(tf.matrix_band_part(self.off_diag_vars, -1, 0),
                                                    tf.zeros([self.nvoxels, self.nparams]),
                                                    name='%s_off_diag_cov_chol' % self.name)

        # Combine diagonal and off-diagonal elements into full matrix
        self.cov_chol = tf.add(tf.matrix_diag(self.std), self.off_diag_cov_chol,
                               name='%s_cov_chol' % self.name)

        # Form the covariance matrix from the chol decomposition
        self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol,
                             name='%s_cov' % self.name)

        self.cov_chol = debug(self, self.cov_chol)
        self.cov = debug(self, self.cov)
        self.mean = debug(self, self.mean)

    def sample(self, nsamples):
        # Use the 'reparameterization trick' to return the samples
        eps = tf.random_normal((self.nvoxels, self.nparams, nsamples), 0, 1, dtype=tf.float32)

        # NB self.cov_chol is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev.
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, self.nparams, 1]),
                             [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.matmul(self.cov_chol, eps), name="%s_sample" % self.name)
        return debug(self, sample)

    def entropy(self):
        det_covar = tf.matrix_determinant(self.cov, name="%s_det" % self.name) # [V]
        entropy = tf.identity(-0.5 * tf.log(det_covar), name="%s_entropy" % self.name)
        return debug(self, entropy)
