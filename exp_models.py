import dist

import tensorflow as tf
import numpy as np

from model import Model
from parameter import Parameter
    
class MultiExpModel(Model):
    """
    Exponential decay with multiple independent decay rates and amplitudes
    """

    def __init__(self, **options):
        Model.__init__(self, **options)
        self.num_exps = options.get("num_exps", 1)
        for n in range(self.num_exps):
            self.params.append(Parameter("amp%i" % (n+1), 
                                         dist.LogNormal(1, 100), 
                                         mean_init=self._init_amp))
            self.params.append(Parameter("r%i" % (n+1), 
                                         dist.LogNormal(1, 100), 
                                         mean_init=0.5))
    
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

    def _init_amp(self, t, data, param):
        if param.name == "amp1":
            return (1.1-self.num_exps*0.1)*tf.reduce_max(data, axis=1)
        else:
            return 0.1*tf.reduce_max(data, axis=1)
        
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
