"""
Definition of prior distribution
"""
import numpy as np

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from .utils import LogBase
from .dist import Normal

PRIOR_TYPE_NONSPATIAL = "N"
PRIOR_TYPE_SPATIAL_MRF = "M"

def get_prior(param, data_model, **kwargs):
    """
    Factory method to return a vertexwise prior
    """
    prior = None
    if isinstance(param.prior_dist, Normal):
        if param.prior_type == "N":
            prior = NormalPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "M":
            prior = MRFSpatialPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "M2":
            prior = MRF2SpatialPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "Mfab":
            prior = FabberMRFSpatialPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "A":
            prior = ARDPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)

    if prior is not None:
        return prior
    else:
        raise ValueError("Can't create prior type %s for distribution %s - unrecognized combination" % (param.prior_type, param.prior_dist))

class Prior(LogBase):
    """
    Base class for a prior, defining methods that must be implemented
    """

    def mean_log_pdf(self, samples):
        """
        :param samples: A tensor of shape [W, P, S] where W is the number
                        of parameter vertices, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples

        :return: A tensor of shape [W] where W is the number of parameter vertices
                 containing the mean log PDF of the parameter samples
                 provided
        """
        raise NotImplementedError()

    def log_det_cov(self):
        raise NotImplementedError()

class NormalPrior(Prior):
    """
    Prior based on a vertexwise univariate normal distribution
    """

    def __init__(self, nvertices, mean, var, **kwargs):
        """
        :param mean: Prior mean value
        :param var: Prior variance
        """
        Prior.__init__(self)
        self.name = kwargs.get("name", "NormalPrior")
        self.nvertices = nvertices
        self.scalar_mean = mean
        self.scalar_var = var
        self.mean = tf.fill([nvertices], mean, name="%s_mean" % self.name)
        self.var = tf.fill([nvertices], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

    def mean_log_pdf(self, samples):
        """
        Mean log PDF for normal distribution

        Note that ``term1`` is a constant offset when the prior variance is fixed and hence
        in earlier versions of the code this was neglected, along with other constant offsets
        such as factors of pi. However when this code is inherited by spatial priors and ARD
        the variance is no longer fixed and this term must be included.
        """
        dx = tf.subtract(samples, tf.reshape(self.mean, [self.nvertices, 1, 1])) # [W, 1, N]
        z = tf.div(tf.square(dx), tf.reshape(self.var, [self.nvertices, 1, 1])) # [W, 1, N]
        term1 = self.log_tf(-0.5*tf.log(tf.reshape(self.var, [self.nvertices, 1, 1])), name="term1")
        term2 = self.log_tf(-0.5*z, name="term2")
        log_pdf = term1 + term2 # [W, 1, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nvertices]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "Non-spatial prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

class FabberMRFSpatialPrior(NormalPrior):
    """
    Prior designed to mimic the 'M' type spatial prior in Fabber.
    
    Note that this uses update equations for ak which is not in the spirit of the stochastic
    method. 'Native' SVB MRF spatial priors are also defined which simply treat the spatial
    precision parameter as an inference variable.

    This code has been verified to generate the same ak estimate given the same input as
    Fabber, however in practice it does not optimize to the same value. We don't yet know
    why.
    """

    def __init__(self, nvertices, mean, var, idx=None, post=None, nn=None, n2=None, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the prior mean at each parameter vertex
        :param var: Tensor of shape [W] containing the prior variance at each parameter vertex
        :param post: Posterior instance
        :param nn: Sparse tensor of shape [W, W] containing nearest neighbour lists
        :param n2: Sparse tensor of shape [W, W] containing second nearest neighbour lists
        """
        NormalPrior.__init__(self, nvertices, mean, var, name="FabberMRFSpatialPrior")
        self.idx = idx

        # Save the original vertexwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour vertices
        self.fixed_mean = self.mean
        self.fixed_var = self.var

        # nn and n2 are sparse tensors of shape [W, W]. If nn[A, B] = 1 then A is
        # a nearest neighbour of B, and similarly for n2 and second nearest neighbours
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

        self.sigmaK = self.log_tf(tf.matrix_diag_part(post.cov)[:, self.idx], name="sigmak") # [W]
        self.wK = self.log_tf(post.mean[:, self.idx], name="wk") # [W]
        self.num_nn = self.log_tf(tf.sparse_reduce_sum(self.nn, axis=1), name="num_nn") # [W]

        # Sum over vertices of parameter variance multiplied by number of 
        # nearest neighbours for each vertex
        trace_term = self.log_tf(tf.reduce_sum(self.sigmaK * self.num_nn), name="trace") # [1]

        # Sum of nearest and next-nearest neighbour mean values
        self.sum_means_nn = self.log_tf(tf.reshape(tf.sparse_tensor_dense_matmul(self.nn, tf.reshape(self.wK, (-1, 1))), (-1,)), name="wksum") # [W]
        self.sum_means_n2 = self.log_tf(tf.reshape(tf.sparse_tensor_dense_matmul(self.n2, tf.reshape(self.wK, (-1, 1))), (-1,)), name="contrib8") # [W]
        
        # vertex parameter mean multipled by number of nearest neighbours
        wknn = self.log_tf(self.wK * self.num_nn, name="wknn") # [W]

        swk = self.log_tf(wknn - self.sum_means_nn, name="swk") # [W]

        term2 = self.log_tf(tf.reduce_sum(swk * self.wK), name="term2") # [1]

        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hk = tf.multiply(tf.to_float(self.nvertices), 0.5) + 1.0
        self.ak = self.log_tf(tf.identity(gk * hk, name="ak"))

    def _setup_mean_var(self, post, nn, n2):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = self.log_tf(8*self.sum_means_nn, name="contrib_nn") # [W]
        contrib_n2 = self.log_tf(-self.sum_means_n2, name="contrib_n2") # [W]
        
        spatial_mean = self.log_tf(contrib_nn / (8*self.num_nn), name="spatial_mean")
        spatial_prec = self.log_tf(self.num_nn * self.ak, name="spatial_prec")

        self.var = self.log_tf(1 / (1/self.fixed_var + spatial_prec), name="%s_var" % self.name)
        #self.var = self.fixed_var
        self.mean = self.log_tf(self.var * spatial_prec * spatial_mean, name="%s_mean" % self.name)
        #self.mean = self.fixed_mean + self.ak

class MRFSpatialPrior(Prior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring vertices using the Markov Random Field method

    This uses the same formalism as the Fabber 'M' type spatial prior but treats the ak
    as a parameter of the optimization.
    """

    def __init__(self, nvertices, mean, var, idx=None, post=None, nn=None, n2=None, **kwargs):
        Prior.__init__(self)
        self.name = kwargs.get("name", "MRFSpatialPrior")
        self.nvertices = nvertices
        self.mean = tf.fill([nvertices], mean, name="%s_mean" % self.name)
        self.var = tf.fill([nvertices], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        # nn is a sparse tensor of shape [W, W]. If nn[A, B] = 1 then A is
        # a nearest neighbour of B
        self.nn = nn

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        # We infer the log of ak.
        self.logak = tf.Variable(-5.0, name="log_ak", dtype=tf.float32)
        self.ak = self.log_tf(tf.exp(self.logak, name="ak"))

    def mean_log_pdf(self, samples):
        r"""
        mean log PDF for the MRF spatial prior.

        This is calculating:

        :math:`\log P = \frac{1}{2} \log \phi - \frac{\phi}{2}\underline{x^T} D \underline{x}`
        """
        samples = tf.reshape(samples, (self.nvertices, -1)) # [W, N]
        self.num_nn = self.log_tf(tf.sparse_reduce_sum(self.nn, axis=1), name="num_nn") # [W]
        dx_diag = self.log_tf(tf.reshape(self.num_nn, (self.nvertices, 1)) * samples, name="dx_diag") # [W, N]
        dx_offdiag = self.log_tf(tf.sparse_tensor_dense_matmul(self.nn, samples), name="dx_offdiag") # [W, N]
        self.dx = self.log_tf(dx_diag - dx_offdiag, name="dx") # [W, N]
        self.xdx = self.log_tf(samples * self.dx, name="xdx") # [W, N]
        term1 = tf.identity(0.5*self.logak, name="term1")
        term2 = tf.identity(-0.5*self.ak*self.xdx, name="term2")
        log_pdf = term1 + term2  # [W, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nvertices]) # [W]

        # Gamma prior if we care
        #q1, q2 = 1, 100
        #mean_log_pdf += (q1-1) * self.logak - self.ak / q2

        return mean_log_pdf

    def __str__(self):
        return "MRF spatial prior"

class ARDPrior(NormalPrior):
    """
    Automatic Relevance Determination prior
    """
    def __init__(self, nvertices, mean, var, **kwargs):
        NormalPrior.__init__(self, nvertices, mean, var, **kwargs)
        self.name = kwargs.get("name", "ARDPrior")
        self.fixed_var = self.var
        
        # Set up inferred precision parameter phi
        self.logphi = tf.Variable(tf.log(1/self.fixed_var), name="log_phi", dtype=tf.float32)
        self.phi = self.log_tf(tf.exp(self.logphi, name="phi"))
        self.var = 1/self.phi
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

    def __str__(self):
        return "ARD prior"

class MRF2SpatialPrior(Prior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring vertices using the Markov Random Field method

    This uses the same formalism as the Fabber 'M' type spatial prior but treats the ak
    as a parameter of the optimization. It differs from MRFSpatialPrior by using the
    PDF formulation of the PDF rather than the matrix formulation (the two are equivalent
    but currently we keep both around for checking that they really are!)

    FIXME currently this does not work unless sample size=1
    """

    def __init__(self, nvertices, mean, var, idx=None, post=None, nn=None, n2=None, **kwargs):
        Prior.__init__(self)
        self.name = kwargs.get("name", "MRF2SpatialPrior")
        self.nvertices = nvertices
        self.mean = tf.fill([nvertices], mean, name="%s_mean" % self.name)
        self.var = tf.fill([nvertices], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        # nn is a sparse tensor of shape [W, W]. If nn[A, B] = 1 then A is
        # a nearest neighbour of B
        self.nn = nn

        # We need the number of samples to implement the log PDF function
        self.sample_size = kwargs.get("sample_size", 5)

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        self.logak = tf.Variable(-5.0, name="log_ak", dtype=tf.float32)
        self.ak = self.log_tf(tf.exp(self.logak, name="ak"))

    def mean_log_pdf(self, samples):
        samples = tf.reshape(samples, (self.nvertices, -1)) # [W, N]
        self.num_nn = self.log_tf(tf.sparse_reduce_sum(self.nn, axis=1), name="num_nn") # [W]

        expanded_nn = tf.sparse_concat(2, [tf.sparse.reshape(self.nn, (self.nvertices, self.nvertices, 1))] * self.sample_size)
        xj = expanded_nn * tf.reshape(samples, (self.nvertices, 1, -1))
        #xi = tf.reshape(tf.sparse.to_dense(tf.sparse.reorder(self.nn)), (self.nvertices, self.nvertices, 1)) * tf.reshape(samples, (1, self.nvertices, -1))
        xi = expanded_nn * tf.reshape(samples, (1, self.nvertices, -1))
        #xi = tf.sparse.transpose(xj, perm=(1, 0, 2)) 
        neg_xi = tf.SparseTensor(xi.indices, -xi.values, dense_shape=xi.dense_shape )
        dx2 = tf.square(tf.sparse.add(xj, neg_xi), name="dx2")
        sdx = tf.sparse.reduce_sum(dx2, axis=0) # [W, N]
        term1 = tf.identity(0.5*self.logak, name="term1")
        term2 = tf.identity(-self.ak * sdx / 4, name="term2")
        log_pdf = term1 + term2  # [W, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nvertices]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "MRF2 spatial prior"

class ConstantMRFSpatialPrior(NormalPrior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring vertices using the Markov Random Field method

    This is equivalent to the Fabber 'M' type spatial prior
    """

    def __init__(self, nvertices, mean, var, idx=None, nn=None, n2=None, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the prior mean at each parameter vertex
        :param var: Tensor of shape [W] containing the prior variance at each parameter vertex
        :param post: Posterior instance
        :param nn: Sparse tensor of shape [W, W] containing nearest neighbour lists
        :param n2: Sparse tensor of shape [W, W] containing second nearest neighbour lists
        """
        NormalPrior.__init__(self, nvertices, mean, var, name="MRFSpatialPrior")
        self.idx = idx

        # Save the original vertexwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour vertices
        self.fixed_mean = self.mean
        self.fixed_var = self.var

        # nn and n2 are sparse tensors of shape [W, W]. If nn[A, B] = 1 then A is
        # a nearest neighbour of B, and similarly for n2 and second nearest neighbours
        self.nn = nn
        self.n2 = n2

    def __str__(self):
        return "Spatial MRF prior (%f, %f) - const" % (self.scalar_mean, self.scalar_var)

    def update_ak(self, post_mean, post_cov):
        # This is the equivalent of CalculateAk in Fabber
        #
        # Some of this could probably be better done using linalg
        # operations but bear in mind this is one parameter only

        self.sigmaK = post_cov[:, self.idx, self.idx] # [W]
        self.wK = post_mean[:, self.idx] # [W]
        self.num_nn = np.sum(self.nn, axis=1) # [W]

        # Sum over vertices of parameter variance multiplied by number of 
        # nearest neighbours for each vertex
        trace_term = np.sum(self.sigmaK * self.num_nn) # [1]

        # Sum of nearest and next-nearest neighbour mean values
        self.sum_means_nn = np.matmul(self.nn, np.reshape(self.wK, (-1, 1))) # [W]
        self.sum_means_n2 = np.matmul(self.n2, tf.reshape(self.wK, (-1, 1))) # [W]
        
        # vertex parameter mean multipled by number of nearest neighbours
        wknn = self.wK * self.num_nn # [W]

        swk = wknn - self.sum_means_nn # [W]

        term2 = np.sum(swk * self.wK) # [1]

        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hk = float(self.nvertices) * 0.5 + 1.0
        self.ak = gk * hk
        self.log.info("%s: ak=%f", self.name, self.ak)

    def _setup_mean_var(self, post_mean, post_cov):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = self.log_tf(8*self.sum_means_nn, name="contrib_nn") # [W]
        contrib_n2 = self.log_tf(-self.sum_means_n2, name="contrib_n2") # [W]
        
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
        self.nvertices = priors[0].nvertices

        # Define a diagonal covariance matrix for convenience
        self.cov = tf.matrix_diag(self.var, name='%s_cov' % self.name)

    def mean_log_pdf(self, samples):
        nvertices = tf.shape(samples)[0]

        mean_log_pdf = tf.zeros([nvertices], dtype=tf.float32)
        for idx, prior in enumerate(self.priors):
            param_samples = tf.slice(samples, [0, idx, 0], [-1, 1, -1])
            param_logpdf = prior.mean_log_pdf(param_samples)
            mean_log_pdf = tf.add(mean_log_pdf, param_logpdf)
        return mean_log_pdf
    
    def log_det_cov(self):
        """
        Determinant of diagonal matrix is product of diagonal entries
        """
        return tf.reduce_sum(tf.log(self.var), axis=1, name='%s_log_det_cov' % self.name)
