"""
Inference forward models for ASL data
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from svb.utils import NP_DTYPE
import numpy as np

from svb import __version__
from svb.model import Model, ModelOption, ValueList, assert_param_overrides_used
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
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=[1]),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float, default=0),
        ModelOption("inferart", "Infer arterial component", type=bool),
        ModelOption("infert1", "Infer T1 value", type=bool),
        ModelOption("pc", "Blood/tissue partition coefficient", type=float, default=0.9),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        if self.plds is not None:
            self.tis = [self.tau + pld for pld in self.plds]
        
        if self.tis is None and self.plds is None:
            raise ValueError("Either TIs or PLDs must be given")

        if self.attsd is None:
            self.attsd = 1.0 if len(self.tis) > 1 else 0.1

        # If repeats is just a single number, cast to list 
        if isinstance(self.repeats, (list, np.ndarray)): 
            if len(self.repeats) == 1:
                self.repeats = self.repeats[0]
            elif ((len(self.repeats) > 1)
                 and any([ r != self.repeats[0] for r in self.repeats ])):
                raise NotImplementedError("Variable repeats for TIs/PLDs")

        # Add the "data_space" to the kwarg options. This is set by 
        # the data_model, "voxel" means voxelwise inference, "node" means
        # surface inference. Noise is ALWAYS defined in "voxel" however. 
        options["data_space"] = self.data_space

        self.params = [
            get_parameter("ftiss", dist="FoldedNormal", 
                          mean=5.0, prior_var=1e6, post_var=1.0, 
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
        if self.inferart:
            self.leadscale = 0.01
            self.params.append(
                get_parameter("fblood", dist="FoldedNormal",
                              mean=0.0, prior_var=1e6, post_var=1.0,
                              post_init=self._init_fblood,
                              prior_type="A",
                              **options)
            )
            self.params.append(
                get_parameter("deltblood", dist="FoldedNormal", 
                              mean=self.att - 0.3, var=self.attsd**2,
                              **options)
            )

        assert_param_overrides_used(self.params, options)

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

        opt_param_idx = 2
        if self.infert1:
            t1 = self.log_tf(params[opt_param_idx], name="t1", shape=True)
            opt_param_idx += 1
        else:
            t1 = self.t1

        if self.inferart:
            fblood = params[opt_param_idx]
            deltblood = params[opt_param_idx+1]
            opt_param_idx += 2
        else:
            fblood = 0
            deltblood = delt

        signal = self.log_tf(self._tissue_signal(t, ftiss, delt, t1), name="tiss_signal")

        if self.inferart:
            signal += self.log_tf(self._art_signal(t, fblood, deltblood), name="art_signal")

        return self.log_tf(signal, name="asl_signal")

    def _tissue_signal(self, t, ftiss, delt, t1):
        """
        PASL/pCASL kinetic model for tissue
        """
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

        return ftiss*signal

    def _art_signal(self, t, fblood, deltblood):
        """
        PASL/pCASL Kinetic model for arterial curve
        
        To avoid problems with the discontinuous gradient at ti=deltblood
        and ti=deltblood+taub, we smooth the transition at these points
        using a Gaussian convolved step function. The sigma value could
        be exposed as a parameter (small value = less smoothing). This is
        similar to the effect of Gaussian dispersion, but can be computed
        without numerical integration
        """
        if self.casl:
            kcblood = 2 * tf.exp(-deltblood / self.t1b)
        else:
            kcblood = 2 * tf.exp(-t / self.t1b)

        # Boolean masks indicating which voxel-timepoints are in the leadin phase
        # and which in the leadout
        leadout = tf.greater(t, tf.add(deltblood, self.tau/2))
        leadin = self.log_tf(tf.logical_not(leadout), name="leadin1", shape=True)

        # If deltblood is smaller than the lead in scale, we could 'lose' some
        # of the bolus, so reduce degree of lead in as deltblood -> 0. We
        # don't really need it in this case anyway since there will be no
        # gradient discontinuity
        leadscale = tf.minimum(deltblood, self.leadscale)
        leadin = self.log_tf(tf.logical_and(leadin, tf.greater(leadscale, 0)), shape=True)

        # Calculate lead-in and lead-out signals
        leadin_signal = self.log_tf(kcblood * 0.5 * (1 + tf.math.erf((t - deltblood) / leadscale)), name="leadin_signal", shape=True)
        leadout_signal = kcblood * 0.5 * (1 + tf.math.erf(-(t - deltblood - self.tau) / self.leadscale))

        # Form final signal from combination of lead in and lead out signals
        signal = tf.zeros(tf.shape(leadin_signal))
        signal = tf.where(leadin, leadin_signal, signal)
        signal = tf.where(leadout, leadout_signal, signal)

        return fblood*signal

    def tpts(self):
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis)*self.repeats, self.data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        if self.slicedt > 0:
            # Generate voxelwise timings array using the slicedt value
            t = np.zeros(list(self.data_model.shape) + [self.data_model.n_tpts])
            for z in range(self.data_model.shape[2]):
                t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))
            t = t.reshape(-1, self.data_model.n_tpts)
            if self.data_model.n_vertices != self.data_model.n_unmasked_voxels:
                # FIXME we don't have voxels_to_vertices currently...
                #t = self.data_model.voxels_to_vertices(t)
                self.log.warn("Ignoring slicedt as vertices!=voxels")
                t = np.array(sum([[ti] * self.repeats for ti in self.tis], []))
        else:
            # Timings are the same for all voxels
            t = np.array(sum([[ti] * self.repeats for ti in self.tis], []))
        return t.astype(NP_DTYPE)

    def __str__(self):
        return "ASL resting state model: %s" % __version__

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """
        return tf.reduce_mean(data, axis=1), None

    def _init_fblood(self, _param, _t, data):
        """
        Initial value for the fblood parameter
        """
        return tf.reduce_mean(data, axis=1), None
