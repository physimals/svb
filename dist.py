"""
Distributions that can be applied to a model parameter
"""
import math

import numpy as np
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

    def __init__(self, mean, var=None, sd=None, **kwargs):
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
        self._geom = kwargs.get("geom", False)

    @property
    def nmean(self):
        if self._geom:
            return math.log(self.mean)
        else:
            # See https://uk.mathworks.com/help/stats/lognstat.html
            return math.log(self.mean**2/math.sqrt(self.var + self.mean**2))

    @property
    def nvar(self):
        if self._geom:
            return math.log(self.var)
        else:
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

class MVN:
    """
    Multivariate Normal Distribution

    This class models a voxelwise multivariate normal distribution 
    (Gaussian) with a set of parameter means and full covariance 
    matrix at each voxel. It uses the reparameterization 'trick' to 
    return deterministic samples from a pre-computed random (0, 1) 
    Gaussian.

    Attributes:
      - mean: tf.Tensor containing the mean value for each parameter (VxP)
      - cov: tf.Tensor containing the covariance matrix (VxPxP)
    """

    def __init__(self, params, mean_init, var_init, covar_init=None, **kwargs):
        """
        Set up basic tensors for the mean and covariance matrices
        
        If a previous run is given it is used to initialize the posterior, otherwise
        the model is given the opportunity to set the initial posterior based on the
        data.

        The posterior is always determined in terms of the Cholesy decomposition - this
        ensures the covariance matrix is positive definite. This means there are always
        two tensors defined in the graph ``log_diag_chol_mp_covar`` which is the log of
        the diagonal of the matrix and ``off_diag_chol_mp_covar`` which are the off-diagnoal
        elements. 
        
        If ``mode_corr`` is ``NO_POST_CORR`` then the latter is a constant
        containing zeros which plays no role in the inference - it exists only so that
        a subsequent run with correlation enabled can be performed using the non-correlated
        run as initialization.
        """
        self.params = params
        self.nparams = len(params)
        self.nvoxels = tf.shape(mean_init)[0]
        self.draw_size = kwargs.get("draw_size", None)
        self.debug = kwargs.get("debug", False)
        self.corr = kwargs.get("corr", False)
        self.name = kwargs.get("name", "MVN")

        # Stack initialized posterior mean and log-variance and transpose
        # so dimensions are (VxP) not (PxV)
        self.mean = tf.Variable(mean_init, validate_shape=False, name="%s_mean" % self.name)
        log_var = tf.Variable(tf.log(var_init), validate_shape=False, name="%s_log_var" % self.name)
        
        if self.corr:
            # Infer a full covariance matrix with on and off-diagonal elements
            off_diag_cov_chol = tf.Variable(covar_init, validate_shape=False, name='%s_off_diag_cov_chol' % self.name)
            
            # Combine diagonal and off-diagonal elements into full matrix
            self.cov_chol = tf.add(tf.matrix_diag(tf.exp(log_var)), tf.matrix_band_part(off_diag_cov_chol, -1, 0), name='%s_cov_chol' % self.name)
        else:     
            # Define this constant in case we want to later use this to initialize a full correlation run
            tf.constant(np.zeros([self.nparams, self.nparams], dtype=np.float32), name='%s_off_diag_cov_chol' % self.name)
        
            # Infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            self.cov_chol = tf.matrix_diag(tf.exp(log_var), name='cov_chol')    
        
        # Form the covariance matrix from the chol decomposition
        self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol, name='%s_cov' % self.name)
    
        # For preventing singular covariance matrix
        self.reg_cov = 1e-6 * tf.constant(np.identity(self.nparams, dtype=np.float32), name="%s_reg_cov" % self.name)

        # Define tensor containing output means in transformed model space
        model_means = []
        for idx, param in enumerate(self.params):
            model_means.append(param.prior.tomodel(self.mean[:, idx]))
        tf.stack(model_means, name="%s_mean_model" % self.name)
         
        if self.debug:
            self.cov_chol = tf.Print(self.cov_chol, [self.cov_chol], "\n%s covariance (Chol decomp)" % self.name, summarize=100)
            self.cov = tf.Print(self.cov, [self.cov], "\n%s covariance" % self.name, summarize=100)
            self.mean = tf.Print(self.mean, [self.mean], "\n%s mean" % self.name, summarize=100)
  
    def sample(self, batch_size):
        """
        :return: A tensor of shape V x P x B where V is the number
                 of voxelse, P is the size of the MVN and B is the
                 batch size
        """
        if self.draw_size is None:
            draw_size = batch_size
        else:
            draw_size = self.draw_size
        eps = tf.random_normal((self.nvoxels, self.nparams, draw_size), 0, 1, dtype=tf.float32)

        # This seems to assume that draw_size is a factor of batch_size? If so, we end
        # up with batch_size samples for each parameter but only draw_size unique samples
        # FIXME is floor division right here?
        ntile = np.floor_divide(batch_size, draw_size)
        eps = tf.tile(eps, [1, 1, ntile])
               
        # NB self.cov_chol is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev. 
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nvoxels, self.nparams, 1]),[1, 1, batch_size])      
        sample = tf.add(tiled_mean, tf.matmul(self.cov_chol, eps))              
        if self.debug:
            sample = tf.Print(sample, [tf.shape(sample), sample], "\nsample", summarize=100)
        return sample

    def mean_log_pdf(self, samples, mask=None):
        """
        :return: A tensor of shape V where V is the number of voxels
                 containing the mean log PDF of the parameter samples
                 provided
        """
        batch_size = tf.shape(samples)[2]

        if mask is not None:
            cov = tf.boolean_mask(self.cov, mask)
            mean = tf.boolean_mask(self.mean, mask)
        else:
            cov = self.cov
            mean = self.mean

        nvoxels = tf.shape(mean)[0]

        det_covar = tf.matrix_determinant(cov, name="%s_det" % self.name) # [NV]
        inv_covar = tf.matrix_inverse(cov, name="%s_inv" % self.name) # [NV, P, P]
        
        dx = tf.subtract(samples, tf.expand_dims(mean, axis=-1)) # [NV x P x B]
        #dx = tf.Print(dx, [tf.shape(dx)], "dx", summarize=10)

        dx = tf.expand_dims(tf.transpose(dx, [0, 2, 1]), axis=2, name="%s_dx" % self.name) # [NV x B x 1 x P]
        #dx = tf.Print(dx, [tf.shape(dx), nvoxels, batch_size, self.nparams], "%s dx shape" % self.name, summarize=10)

        dxt = tf.reshape(dx, [nvoxels, batch_size, self.nparams, 1], name="%s_dxt" % self.name) # [NV x B x P x 1]
        #dxt = tf.Print(dxt, [tf.shape(dxt)], "\n%s dxt shape" % self.name, summarize=100)

        inv_covar_tile = tf.tile(tf.reshape(inv_covar, [nvoxels, 1, self.nparams, self.nparams]), [1, batch_size, 1, 1]) # [NV x B x P x P]
        #inv_covar_tile = tf.Print(inv_covar_tile, [tf.shape(inv_covar_tile)], "inv_covar_tile", summarize=10)

        mul1 = tf.matmul(inv_covar_tile, dxt) # [NV x B x P x 1]
        mul2 = tf.matmul(dx, mul1, name="%s_mul" % self.name) # [NV x B x 1 x 1]
        #mul2 = tf.Print(mul2, [mul2[95]], "mul2", summarize=100)

        pdf = -0.5*(tf.tile(tf.reshape(tf.log(det_covar), [nvoxels, 1]), [1, batch_size]) + tf.reshape(mul2, [nvoxels, batch_size]))
        #pdf = tf.Print(pdf, [tf.reduce_mean(pdf, axis=1)[95]], "\n%s_pdf" % name, summarize=100)
        return tf.reduce_mean(pdf, axis=1)

class ConstantMVN(MVN):

    def __init__(self, params, mean_init, var_init, covar_init=None, **kwargs):
        """
        Set up basic tensors for the mean and covariance matrices
        
        If a previous run is given it is used to initialize the posterior, otherwise
        the model is given the opportunity to set the initial posterior based on the
        data.

        The posterior is always determined in terms of the Cholesy decomposition - this
        ensures the covariance matrix is positive definite. This means there are always
        two tensors defined in the graph ``log_diag_chol_mp_covar`` which is the log of
        the diagonal of the matrix and ``off_diag_chol_mp_covar`` which are the off-diagnoal
        elements. 
        
        If ``mode_corr`` is ``NO_POST_CORR`` then the latter is a constant
        containing zeros which plays no role in the inference - it exists only so that
        a subsequent run with correlation enabled can be performed using the non-correlated
        run as initialization.
        """
        self.name = kwargs.get("name", "constantMVN")
        self.params = params
        self.nparams = len(params)
        self.nvoxels = tf.shape(mean_init)[0]
        self.draw_size = kwargs.get("draw_size", None)
        self.debug = kwargs.get("debug", False)
        self.corr = kwargs.get("corr", False)

        # Stack initialized posterior mean and log-variance and transpose
        # so dimensions are (VxP) not (PxV)
        self.mean = tf.identity(mean_init, name="%s_mean" % self.name)
        log_var = tf.log(var_init, name="%s_log_var" % self.name)
        #log_var = tf.Print(log_var, [log_var], "Prior log var")

        if self.corr:
            # Infer a full covariance matrix with on and off-diagonal elements
            off_diag_cov_chol = tf.identity(covar_init, name='%s_off_diag_cov_chol' % self.name)
            
            # Combine diagonal and off-diagonal elements into full matrix
            self.cov_chol = tf.add(tf.matrix_diag(tf.sqrt(tf.exp(log_var))), tf.matrix_band_part(off_diag_cov_chol, -1, 0), name='%s_cov_chol' % self.name)
        else:     
            # Define this constant in case we want to later use this to initialize a full correlation run
            tf.constant(np.zeros([self.nparams, self.nparams], dtype=np.float32), name='%s_off_diag_cov_chol' % self.name)
        
            # Infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            self.cov_chol = tf.matrix_diag(tf.sqrt(tf.exp(log_var)), name='%s_cov_chol' % self.name)    
        
        # Form the covariance matrix from the chol decomposition
        self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol, name='%s_cov' % self.name)
    
        # For preventing singular covariance matrix
        self.reg_cov = 1e-6 * tf.constant(np.identity(self.nparams, dtype=np.float32), name="%s_reg_cov" % self.name)
         
        if self.debug:
            self.cov_chol = tf.Print(self.cov_chol, [self.cov_chol], "\n%s covariance (Chol decomp)" % self.name, summarize=100)
            self.cov = tf.Print(self.cov, [self.cov], "\n%s covariance" % self.name, summarize=100)
            self.mean = tf.Print(self.mean, [self.mean], "\n%s mean" % self.name, summarize=100)
  