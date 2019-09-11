"""
Posterior distribution for stochastic VB inference
"""
import tensorflow as tf

from .utils import LogBase

class Parameter(LogBase):
    """
    A model parameter
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
        """
        LogBase.__init__(self)

        custom_vals = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_vals)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.prior = kwargs.get("prior")
        self.post = kwargs.get("post", self.prior)
        self._initialise = kwargs.get("initialise", None)

    def voxelwise_prior(self, nvoxels, **kwargs):
        return self.prior.voxelwise_prior(nvoxels, name=self.name, **kwargs)

    def voxelwise_posterior(self, t, data, **kwargs):
        return self.post.voxelwise_posterior(self, t, data, self._initialise, name=self.name, **kwargs)

class GlobalParameter(Parameter):
    """
    A Parameter which takes the same value at every voxel
    """

    def __init__(self, *args, **kwargs):
        Parameter.__init__(self, *args, **kwargs)

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
        