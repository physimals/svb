"""
Miscellaneous models mostly for testing
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from svb import __version__
from svb.model import Model
from svb.parameter import get_parameter

class ConstantModel(Model):
    """
    Model which generates a constant signal
    """

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        self.params += [
            get_parameter("mu", dist="Normal", mean=0.0,
                          prior_var=1e6, post_var=1.0, 
                          **options),
        ]

    def evaluate(self, params, tpts):
        return params[0] * tf.ones_like(tpts)

    def __str__(self):
        return "Constant signal model: %s" % __version__

class PolyModel(Model):
    """
    Model which generates a signal from a polynomial
    """

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        self._degree = options.get("degree", 2)
        for idx in range(self._degree+1):
            self.params.append(
                get_parameter("c%i" % idx, 
                              dist="Normal", mean=0.0,
                              prior_var=1e6, post_var=1.0, 
                              **options),
            )

    def evaluate(self, params, tpts):
        ret = None
        for idx in range(self._degree):
            c = params[idx]
            contrib = c * tf.pow(tpts, idx)
            if ret is None:
                ret = contrib
            else:
                ret += contrib
        return ret

    def __str__(self):
        return "Polynomial model: %s" % __version__
