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

    def __init__(self, name, **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param prior: Dist instance giving the parameter's prior distribution. Required.
        :param post:  Dist instance giving the initial posterior distribution. If not
                      given, uses prior as initial posterior. 
        :param desc: Optional parameter description
        :param initialise: Optional Callable which will be called to initialise the mean and variance
                           of the posterior from the data
        """
        custom_vals = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_vals)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.debug = kwargs.get("debug", False)
        self.prior = kwargs.get("prior")
        self.post = kwargs.get("post", self.prior)
        self._initialise = kwargs.get("initialise", self._null_init)

    def _null_init(self, *args):
        return None, None

    def _initial_mean(self, t, data):
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
            mean_init = tf.random.normal([nvoxels], self.prior.mean, math.sqrt(self.prior.var), dtype=tf.float32)
        return mean_init

    def _initial_var(self, t, data):
        """
        Get initial variance value for the posterior distribution

        Note that this is the variance for the underlying Gaussian distribution
        of the parameter which will be inferred
        """
        nvoxels = tf.shape(data)[0]
        if self._var_init is not None:
            print(self._var_init, self.prior.var)
            if isinstance(self._var_init, collections.Callable):
                var_init = self._var_init(t, data, self)
            else:
                var_init = tf.fill([nvoxels], self._var_init)
        else:
            # FIXME need principled way to initialize variance
            var_init = tf.truncated_normal([nvoxels], 0.13, 0.06, dtype=tf.float32)
            var_init = tf.Print(var_init, [var_init], "VAR_INIT")
            #var_init = tf.fill([nvoxels], 1)
            print("Default initial var")
        return var_init
        
    def initial(self, t, data):
        """
        Get the posterior mean and variance which will be
        inferred as part of the training process.

        :param t: Timeseries Tensor of shape (1xN) or (VxN)
                  where V is the number of voxels and N is the
                  number of time points.
        :param data: Data Tensor of shape (VxN) where V is the 
                     number of voxels and N is the number of time 
                     points.

        :return: Tuple of two tf.Tensor objects. The first is the
                 initial mean, the second the variance. 
        """
        nvoxels = tf.shape(data)[0]
        initial_mean, initial_var = self._initialise(self, t, data)

        if initial_mean is None:
            print("postmean", self.name, self.post.mean)
            initial_mean = tf.fill([nvoxels], self.post.mean)
        if initial_var is None:
            print("postvar",self.name, self.post.var)
            initial_var = tf.fill([nvoxels], self.post.var)
        return initial_mean, initial_var

class GlobalParameter(Parameter):
    """
    A Parameter which takes the same value at every voxel
    """

    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)

    def initial(self, t, data, mean_init=None, log_var_init=None):
        return tf.reduce_mean(self._initial_mean(t, data)), tf.reduce_mean(self._initial_var(t, data))
        
class RoiParameter(Parameter):
    """
    A parameter which takes a weighted sum value at each voxel, determined
    by a set of partial-volume ROIs
    """

    def __init__(self, *args, **kwargs):
        """
        :param rois: Sequence of arrays of length [nvoxels] containing
                     partial-volume weightings for each ROI
        """
        Parameter.__init__(self, *args, **kwargs)
        self._rois = kwargs["rois"]

    def initial(self, t, data, mean_init=None, log_var_init=None):
        """
        Create a single Variable to initialize the posterior mean
        and log variance and broadcast it across all voxels
        """
        if mean_init is None:
            mean_init = self.mean_init(t, data)

        if log_var_init is None:
            log_var_init = self.log_var_init(t, data)
        
        voxelwise_mean, voxelwise_var = None, None
        for idx, roi in enumerate(self._rois):
            roi = tf.constant(roi, dtype=tf.float32)
            mean = tf.Variable(tf.reduce_mean(mean_init), validate_shape=False, dtype=tf.float32, name="mean_roi_%i" % idx)
            #mean = tf.Print(mean, [mean], "mean_roi_%i" % idx)
            log_var = tf.Variable(tf.reduce_mean(log_var_init), validate_shape=False, dtype=tf.float32, name="log_var_roi_%i" % idx)
            partial_mean = tf.multiply(roi, mean)
            if voxelwise_mean is None:
                voxelwise_mean = partial_mean
            else:
                voxelwise_mean = tf.add(voxelwise_mean, partial_mean)
            # FIXME weighted average correct for log var?
            if voxelwise_var is None:
                voxelwise_var = roi * tf.exp(log_var)
            else:
                voxelwise_var = tf.add(voxelwise_var, roi * tf.exp(log_var))

        return voxelwise_mean, tf.log(voxelwise_var)
        