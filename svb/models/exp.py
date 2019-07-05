"""
Multi-exponential models
"""
import tensorflow as tf

from svb.model import Model
from svb.parameter import Parameter
import svb.dist as dist

class MultiExpModel(Model):
    """
    Exponential decay with multiple independent decay rates and amplitudes
    """

    def __init__(self, **options):
        Model.__init__(self, **options)
        self.num_exps = options.get("num_exps", 1)
        for n in range(self.num_exps):
            self.params += [
                Parameter("amp%i" % (n+1),
                          prior=dist.LogNormal(1.0, 1e6),
                          post=dist.LogNormal(1.0, 1.5),
                          initialise=self._init_amp,
                          **options),
                Parameter("r%i" % (n+1),
                          prior=dist.LogNormal(1.0, 1e6),
                          post=dist.LogNormal(1.0, 1.5),
                          **options),
            ]

    def _init_amp(self, _param, _t, data):
        return tf.reduce_max(data, axis=1) / self.num_exps, None

    def evaluate(self, params, t):
        ret = None
        for n in range(self.num_exps):
            amp = params[2*n]
            r = params[2*n+1]
            contrib = amp * tf.exp(-r * t)
            if ret is None:
                ret = contrib
            else:
                ret += contrib
        return ret

class ExpModel(MultiExpModel):
    """
    Simple exponential decay model
    """
    def __init__(self, **options):
        MultiExpModel.__init__(self, num_exps=1, **options)

class BiExpModel(MultiExpModel):
    """
    Exponential decay with two independent decay rates and amplitudes
    """
    def __init__(self, **options):
        MultiExpModel.__init__(self, num_exps=2, **options)
