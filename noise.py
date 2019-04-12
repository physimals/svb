"""
Noise model
"""

import numpy as np
import tensorflow as tf

from parameter import Parameter
import dist

class NoiseParameter(Parameter):

    def __init__(self):
        Parameter.__init__(self, "noise", prior=dist.Normal(0, 1e6), var_init=None)

    def _initial_mean(self, t, data):
        _, variance = tf.nn.moments(data, axes=[1])
        return tf.log(tf.maximum(variance, 1e-4))

    def log_likelihood(self, data, pred, noise):
        """
        Calculate the log-likelihood of the data

        Note that we are using the log of the noise Gaussian variance as the noise parameter
        here
        """
        nt = tf.to_float(tf.shape(data)[1])

        zscore = tf.square(data - pred) / tf.exp(noise)
        reconstr_loss = tf.reduce_sum(0.5 * (zscore + noise), axis=1, name="reconstr1")
        #reconstr_loss = tf.reduce_mean(0.5 * (sum_squares + noise), axis=1, name="reconstr")
        if self.debug:
            reconstr_loss = tf.Print(reconstr_loss, [reconstr_loss], "\nreconstr1", summarize=100)

        return reconstr_loss
        