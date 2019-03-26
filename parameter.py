"""
Posterior distribution for stochastic VB inference
"""
import collections
import math

import numpy as np
import tensorflow as tf

class Parameter:
    """
    A model parameter
    """

    def __init__(self, name, dist, desc="No description given", **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param dist: Dist instance giving the parameter's prior distribution
        :param desc: Optional parameter description

        Keyword arguments (optional):
         - ``mean_init`` Initial value for the posterior mean either as a numeric
                         value or a callable which takes the parameters t, data, param_name 
         - ``log_var_init`` Initial value for the posterior log variance either as a numeric
                            value or a callable which takes the parameters t, data, param_name 
        """
        self.name = name
        self.desc = desc
        self.dist = dist
        self._mean_init = kwargs.get("mean_init", self.dist.nmean)
        self._log_var_init = kwargs.get("log_var_init", math.log(self.dist.nvar/100))
        #self._log_var_init = kwargs.get("log_var_init", None)

    def prior(self):
        """
        Get prior mean and variance for the parameter in the 
        underlying MVN distribution
        """
        return self.dist.nmean, self.dist.nvar

    def mean_init(self, t, data):
        """
        Get initial mean value for the posterior distribution

        Note that this is the mean for the underlying Gaussian distribution
        of the parameter
        """
        nvoxels = tf.shape(data)[0]
        if self._mean_init is not None:
            if isinstance(self._mean_init, collections.Callable):
                mean_init = self._mean_init(t, data, self)
            else:
                mean_init = tf.fill([nvoxels], self._mean_init)
        else:
            mean_init = tf.random.normal([nvoxels], self.dist.nmean, math.sqrt(self.dist.nvar), dtype=tf.float32)
        return mean_init

    def log_var_init(self, t, data):
        """
        Get initial log variance value for the posterior distribution

        Note that this is the log variance for the underlying Gaussian distribution
        of the parameter
        """
        nvoxels = tf.shape(data)[0]
        if self._log_var_init is not None:
            if isinstance(self._log_var_init, collections.Callable):
                log_var_init = self._log_var_init(t, data, self)
            else:
                log_var_init = tf.fill([nvoxels], self._log_var_init)
        else:
            # FIXME need principled way to initialize variance
            log_var_init = tf.truncated_normal([nvoxels], -2, 0.4, dtype=tf.float32)
            #log_var_init = tf.fill([nvoxels], -2.0)
        return log_var_init
        
    def posterior(self, t, data, mean_init=None, log_var_init=None):
        """
        Get the posterior mean and variance which will be
        inferred as part of the training process.

        :param t: Timeseries Tensor of shape (1xN) or (VxN)
                  where V is the number of voxels and N is the
                  number of time points.
        :param data: Data Tensor of shape (VxN) where V is the 
                     number of voxels and N is the number of time 
                     points.
        :param mean_init: Tensor of shape (V) containing the initial 
                          value of the mean of the posterior at each 
                          voxel.
        :param log_var_init: Tensor of shape (V) containing the initial
                             value of the log variance for the 
                             posterior at each voxel.

        :return: Tuple of two tf.Tensor objects. The first is the
                 posterior mean, the second the log variance. 
                 These should be backed by tf.Variable objects
                 which will be inferred. The exact mapping of
                 Variables to the mean/variance tensors is not
                 specified. By default it is one per voxel but
                 it could equallly be a single variable for
                 all voxels, or one per ROI, etc. Initial
                 values of the posterior may be based on the
                 data supplied.
        """
        nvoxels = tf.shape(data)[0]
        if mean_init is None:
            mean_init = self.mean_init(t, data)

        if log_var_init is None:
            log_var_init = self.log_var_init(t, data)
            
        #mean_init = tf.Print(mean_init, [mean_init], "mean_init_%s" % self.name, summarize=100)
        means = tf.Variable(mean_init, validate_shape=False)
        log_vars = tf.Variable(log_var_init, validate_shape=False)
        return means, log_vars

class GlobalParameter(Parameter):
    """
    A Parameter which takes the same value at every voxel
    """

    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)

    def posterior(self, t, data, mean_init=None, log_var_init=None):
        """
        Create a single Variable to initialize the posterior mean
        and log variance and broadcast it across all voxels
        """
        nvoxels = tf.shape(data)[0]
        if mean_init is None:
            mean_init = self.mean_init(t, data)

        if log_var_init is None:
            log_var_init = self.log_var_init(t, data)
            
        mean = tf.Variable(tf.reduce_mean(mean_init), validate_shape=False)
        log_var = tf.Variable(tf.reduce_mean(log_var_init), validate_shape=False)
        return tf.fill([nvoxels], mean), tf.fill([nvoxels], log_var)
        
