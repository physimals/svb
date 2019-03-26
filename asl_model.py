"""
Inference forward models for ASL data
"""
import math

import tensorflow as tf
import numpy as np

from model import Model, Parameter
import dist

class AslRestModel(Model):
    """
    ASL resting state model

    FIXME integrate with oxasl AslImage class?
    """

    def __init__(self, **options):
        Model.__init__(self, **options)
        self.tau = options["tau"]
        self.casl = options.get("casl", True)
        self.bat = options.get("bat", 1.3)
        self.batsd = options.get("batsd", 1.0)
        self.t1 = options.get("t1", 1.3)
        self.t1b = options.get("t1b", 1.65)
        self.pc = options.get("pc", 0.9)
        self.f_calib = options.get("fcalib", 0.01)

        self.params.append(Parameter("ftiss", dist.LogNormal(1, 1e12, geom=True), mean_init=self._init_flow))
        self.params.append(Parameter("delt", dist.Normal(self.bat, sd=self.batsd)))
    
    def _init_flow(self, t, data, param=None):
        """
        Initial value for the flow parameter
        """
        flow = tf.log(tf.reduce_max(data, axis=1))
        #return tf.Print(flow, [flow], "flow", summarize=100)
        return flow

    def evaluate(self, params, t):
        """
        Basic PASL/pCASL kinetic model
        """
        if self.debug:
            params = tf.Print(params, [params], "\nparams", summarize=100)

        # Extract parameter tensors
        ftiss = params[0]
        delt = params[1]

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = tf.greater(t, tf.add(self.tau, delt))
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))
        if self.debug:
            post_bolus = tf.Print(post_bolus, [post_bolus], "\npost_bolus", summarize=100)
            during_bolus = tf.Print(during_bolus, [during_bolus], "\nduring_bolus", summarize=100)

        # Rate constants
        t1_app = 1 / (1 / self.t1 + self.f_calib / self.pc)
        r = 1 / t1_app - 1 / self.t1b
        f = 2 * tf.exp(-t / t1_app)
        if self.debug:
           t1_app = tf.Print(t1_app, [t1_app], "\nt1_app", summarize=100)

        # Calculate signal
        if self.casl:
            during_bolus_signal = 2 * t1_app * tf.exp(-delt / self.t1b) * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = 2 * t1_app * tf.exp(-delt / self.t1b) * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            during_bolus_signal = f / r * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = f / r * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))
        if self.debug:
            post_bolus_signal = tf.Print(post_bolus_signal, [post_bolus_signal], "\npost_bolus_signal", summarize=100)
            during_bolus_signal = tf.Print(during_bolus_signal, [during_bolus_signal], "\nduring_bolus_signal", summarize=100)

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal))
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)
        if self.debug:
           signal = tf.Print(signal, [signal], "\nsignal", summarize=100)

        return ftiss*signal

