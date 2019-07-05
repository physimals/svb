"""
Inference forward models for ASL data
"""
import tensorflow as tf

import numpy as np

from svb import __version__
from svb.model import Model
from svb.parameter import Parameter
import svb.dist as dist

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
        self.batsd = options.get("batsd", 0.5)
        self.t1 = options.get("t1", 1.3)
        self.t1b = options.get("t1b", 1.65)
        self.pc = options.get("pc", 0.9)
        self.f_calib = options.get("fcalib", 0.01)
        self.slicedt = options.get("slicedt", 0)
        self.repeats = options.get("repeats", 1)
        self.plds = options.get("plds", None)
        self.tis = options.get("tis", None)
        if self.plds is not None:
            self.tis = [self.tau + pld for pld in self.plds]

        #pvgm = options.get("pvgm", None)
        #pvwm = options.get("pvwm", None)
        #if pvgm is not None and pvwm is not None:
        #    pvcsf = np.ones(pvgm.shape, dtype=np.float32) - pvgm - pvwm
        #    self.params.append(RoiParameter("ftiss", dist.FoldedNormal(0, 1e12), mean_init=self._init_flow, log_var_init=0.0, rois=[pvgm, pvwm, pvcsf]))
        #else:
        self.params = [
            Parameter("ftiss",
                      prior=dist.FoldedNormal(0.0, 1e6),
                      post=dist.FoldedNormal(0.0, 2.0),
                      initialise=self._init_flow),
            Parameter("delttiss",
                      prior=dist.FoldedNormal(self.bat, self.batsd**2)),
        ]

    def evaluate(self, params, tpts):
        """
        Basic PASL/pCASL kinetic model

        :param t: Sequence of time values of length N
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is MxN tensor where M is the number of voxels. This
                      may be supplied as a PxMxN tensor where P is the number of
                      parameters.

        :return: MxN tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        # Extract parameter tensors
        t = self.log_tf(tpts, name="tpts", shape=True)
        ftiss = self.log_tf(params[0], name="ftiss", shape=True)
        delt = self.log_tf(params[1], name="delt", shape=True)

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = self.log_tf(tf.greater(t, tf.add(self.tau, delt), name="post_bolus"), shape=True)
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))

        # Rate constants
        t1_app = 1 / (1 / self.t1 + self.f_calib / self.pc)
        r = 1 / t1_app - 1 / self.t1b
        f = 2 * tf.exp(-t / t1_app)

        # Calculate signal
        if self.casl:
            during_bolus_signal = 2 * t1_app * tf.exp(-delt / self.t1b) * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = 2 * t1_app * tf.exp(-delt / self.t1b) * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            during_bolus_signal = f / r * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = f / r * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))

        post_bolus_signal = self.log_tf(post_bolus_signal, name="post_bolus_signal", shape=True)
        during_bolus_signal = self.log_tf(during_bolus_signal, name="during_bolus_signal", shape=True)

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal))
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return ftiss*signal

    def tpts(self, n_tpts, shape):
        if n_tpts != len(self.tis):
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis), n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        if self.slicedt > 0:
            # Generate voxelwise timings array using the slicedt value
            t = np.zeros(shape)
            for z in range(shape[2]):
                t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))
            t = t.reshape(-1, n_tpts)
        else:
            # Timings are the same for all voxels
            t = np.array(sum([[ti] * self.repeats for ti in self.tis], []))
        return t

    def __str__(self):
        return "ASL resting state model: %s" % __version__

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """
        flow = tf.reduce_max(data, axis=1)
        return flow, None
