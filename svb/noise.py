"""
Noise model
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
   
from .parameter import Parameter
from . import dist

class NoiseParameter(Parameter):
    """
    Noise parameter providing Gaussian (white) noise
    """

    def __init__(self, **kwargs):
        Parameter.__init__(self, "noise",
<<<<<<< HEAD
                           prior=dist.LogNormal(1.0, 1e6),
                           post=dist.LogNormal(1.0, 1.5),
=======
                           prior=dist.LogNormal(1.0, 2e5),
                           post=dist.LogNormal(1.0, 1.02),
>>>>>>> master
                           post_init=self._init_noise,
                           **kwargs)

    def _init_noise(self, _param, _t, data):
        data_mean, data_var = tf.nn.moments(tf.constant(data), axes=1)
        return tf.where(tf.equal(data_var, 0), tf.ones_like(data_var), data_var), None

    def log_likelihood(self, data, pred, noise_var, nt):
        """
        Calculate the log-likelihood of the data

        :param data: Tensor of shape [V, B]
        :param pred: Model prediction tensor with shape [V, S, B]
        :param noise: Noise parameter samples tensor with shape [V, S]
        :return: Tensor of shape [V] containing mean log likelihood of the 
                 data at each voxel with respect to the noise parameters
        """
        nvoxels = tf.shape(data)[0]
        batch_size = tf.shape(data)[1]
        sample_size = tf.shape(pred)[1]

        # Possible to get zeros when using surface projection
        noise_var = tf.where(tf.equal(noise_var, 0), tf.ones_like(noise_var), noise_var)
<<<<<<< HEAD
        log_noise_var = self.log_tf(tf.log(noise_var, name="log_noise_var"), force=False)
=======
        log_noise_var = self.log_tf(tf.log(noise_var, name="log_noise_var"))

>>>>>>> master
        data = self.log_tf(tf.tile(tf.reshape(data, [nvoxels, 1, batch_size]), [1, sample_size, 1], name="data"), force=False)
        pred = self.log_tf(pred, force=False)

        # Square_diff has shape [NV, S, B]
        square_diff = self.log_tf(tf.square(data - pred, name="square_diff"), force=False)
        sum_square_diff = self.log_tf(tf.reduce_sum(square_diff, axis=-1), name="ssq", force=False)

        # Since we are processing only a batch of the data at a time, we need to scale the 
        # sum of squared differences term correctly. Note that 'nt' is already the full data
        # size
        scale = self.log_tf(tf.divide(tf.to_float(nt), tf.to_float(batch_size), name="scale"))

        # Log likelihood has shape [NV, S]
        log_likelihood = 0.5 * (log_noise_var * tf.to_float(nt) +
                                scale * sum_square_diff / noise_var)
        log_likelihood = self.log_tf(tf.identity(log_likelihood, name="log_likelihood"), force=False)

        # Mean over samples - reconstr_loss has shape [NV]
        return self.log_tf(tf.reduce_mean(log_likelihood, axis=1, name="mean_log_likelihood"))
