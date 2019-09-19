"""
SVB - Model parameters

This module defines a set of classes of model parameters.

The factory methods which create priors/posteriors can
make use of the instance class to create the appropriate
type of voxelwise prior/posterior
"""
import tensorflow as tf

from .utils import LogBase
from . import prior
from . import posterior

class Parameter(LogBase):
    """
    A standard model parameter
    """

    def __init__(self, name, **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param prior: Dist instance giving the parameter's prior distribution
        :param desc: Optional parameter description

        Keyword arguments (optional):
         - ``mean_init`` Initial value for the posterior mean either as a numeric
                         value or a callable which takes the parameters t, data, param_name
         - ``log_var_init`` Initial value for the posterior log variance either as a numeric
                            value or a callable which takes the parameters t, data, param_name
         - ``param_overrides`` Dictionary keyed by parameter name. Value should be dictionary
                               of keyword arguments which will override those defined as
                               existing keyword arguments
        """
        LogBase.__init__(self)

        custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_kwargs)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.prior_dist = kwargs.get("prior")
        self.post_dist = kwargs.get("post", self.prior_dist)
        self.post_initialise = kwargs.get("initialise", None)
        self.priortype = kwargs.get("priortype", "N")

    def __str__(self):
        return "Parameter: %s" % self.name

class GlobalParameter(Parameter):
    """
    A Parameter which takes the same value at every voxel

    FIXME broken currently. Needs a custom voxelwise prior class
    """

    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)

class RoiParameter(Parameter):
    """
    A parameter which takes a weighted sum value at each voxel, determined
    by a set of partial-volume ROIs

    FIXME broken currently. Needs a custom voxelwise prior class
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
            mean = tf.Variable(tf.reduce_mean(mean_init), validate_shape=False, dtype=tf.float32,
                               name="mean_roi_%i" % idx)
            #mean = tf.Print(mean, [mean], "mean_roi_%i" % idx)
            log_var = tf.Variable(tf.reduce_mean(log_var_init), validate_shape=False, dtype=tf.float32,
                                  name="log_var_roi_%i" % idx)
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
        