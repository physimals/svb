"""
Inference forward models for ASL data
"""
import tensorflow as tf

import numpy as np

from svb import __version__
from svb.model import Model, ModelOption, ValueList
from svb.parameter import get_parameter
import svb.dist as dist
import svb.prior as prior

class AslRestModel(Model):
    """
    ASL resting state model

    FIXME integrate with oxasl AslImage class?
    """

    OPTIONS = [
        ModelOption("tau", "Bolus duration", units="s", clargs=("--tau", "--bolus"), type=float, default=1.8),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=False),
        ModelOption("att", "Bolus arrival time", units="s", type=float, default=1.3),
        ModelOption("attsd", "Bolus arrival time prior std.dev.", units="s", type=float, default=None),
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=1.65),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption("plds", "Post-labelling delays (for CASL instead of TIs)", units="s", type=ValueList(float)),
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=1),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float),
        ModelOption("infert1", "Infer T1 value", type=bool),
        ModelOption("pc", "Blood/tissue partition coefficient", type=float, default=0.9),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
    ]

    def __init__(self, **options):
        Model.__init__(self, **options)
        if self.plds is not None:
            self.tis = [self.tau + pld for pld in self.plds]
        if self.attsd is None:
            self.attsd = 1.0 if len(self.tis) > 1 else 0.1
        if len(self.repeats) == 1:
            # FIXME variable repeats
            self.repeats = self.repeats[0]

        self.params = [
            get_parameter("ftiss", dist="FoldedNormal", 
                          mean=0.0, prior_var=1e6, post_var=1.0, 
                          post_init=self._init_flow,
                          **options),
            get_parameter("delttiss", dist="FoldedNormal", 
                          mean=self.att, var=self.attsd**2,
                          **options)
        ]
        if self.infert1:
            self.params.append(
                get_parameter("t1", mean=1.3, var=0.01, **options)
            )

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

        if self.infert1:
            t1 = self.log_tf(params[2], name="t1", shape=True)
        else:
            t1 = self.t1

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = self.log_tf(tf.greater(t, tf.add(self.tau, delt), name="post_bolus"), shape=True)
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))

        # Rate constants
        t1_app = 1 / (1 / t1 + self.fcalib / self.pc)

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = 2 * t1_app * tf.exp(-delt / self.t1b)
            during_bolus_signal =  factor * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = factor * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-t / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = factor * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))

        post_bolus_signal = self.log_tf(post_bolus_signal, name="post_bolus_signal", shape=True)
        during_bolus_signal = self.log_tf(during_bolus_signal, name="during_bolus_signal", shape=True)

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal))
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return self.log_tf(ftiss*signal, name="asl_signal", shape=True)

    def tpts(self, data_model):
        if data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis)*self.repeats, data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        if self.slicedt > 0:
            # Generate voxelwise timings array using the slicedt value
            t = np.zeros(list(data_model.shape) + [data_model.n_tpts])
            for z in range(data_model.shape[2]):
                t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))
        else:
            # Timings are the same for all voxels
            t = np.array(sum([[ti] * self.repeats for ti in self.tis], []))
        return t.reshape(-1, data_model.n_tpts)

    def __str__(self):
        return "ASL resting state model: %s" % __version__

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """
        flow = tf.reduce_max(data, axis=1)
        return flow, None
