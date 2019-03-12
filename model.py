import tensorflow as tf

class Parameter:
    """
    A model parameter
    """

    def __init__(self, name, desc="No description given", **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param desc: Optional parameter description

        Keyword arguments:
         - ``prior_mean`` Built in prior mean value (default: 0)
         - ``prior_var`` Built in prior variance (default: 100)
         - ``prior_sd`` Alternative to ``prior_var`` - built in prior standard deviation
        """
        self.name = name
        self.desc = desc
        self.prior_mean = kwargs.get("prior_mean", 0)
        if "prior_var" in kwargs and "prior_sd" in kwargs:
            raise ValueError("Can't specify prior standard deviation and variance at the same time for parameter %s" % name)
        elif "prior_sd" in kwargs:
            self.prior_var = kwargs["prior_sd"]**2
        else:
            self.prior_var = kwargs.get("prior_var", 100)
    
class Model:
    """
    A forward model
    
    :attr params: Sequence of ``Parameter`` objects
    :attr nparams: Number of model parameters
    """

    def __init__(self, options):
        self.params = []

    @property
    def nparams(self):
        """
        Number of parameters in the model
        """
        return len(self.params)

    def param_idx(self, name):
        """
        :return: the index of a named parameter
        """
        for idx, param in enumerate(self.params):
            if param.name == name:
                return idx
        raise ValueError("Parameter not found in model: %s" % name)
        
    def evaluate(self, params, t):
        """
        Evaluate the model
        
        :param t: Sequence of time values
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is a Numpy array with the same length as ``t``

        :return: Numpy array of same length as ``t`` containing model output
                 at the corresponding value of ``t`` and corresponding parameter
                 values.
        """
        raise NotImplementedError("evaluate")

    def ievaluate(self, params, t):
        with tf.Session():
            return self.evaluate(params, t).eval()

    def test_data(self, t, params_map):
        param_values = [params_map[param.name] for param in self.params]
        with tf.Session():
            clean = self.evaluate(param_values, t).eval()
            if "noise_sd" in params_map:
                import numpy as np
                np.random.seed(1)
                noisy = np.random.normal(clean, params_map["noise_sd"])
                return clean, noisy
            else:
                return clean

    def update_initial_posterior(self, t, data, post):
        """
        Update the initial posterior mean values

        This is an optional method
        """
        pass

class ExpModel(Model):
    
    def __init__(self, options):
        Model.__init__(self, options)
        self.params.append(Parameter("amp1", prior_mean=0, prior_var=100))
        self.params.append(Parameter("r1", prior_mean=0, prior_var=100))
    
    def evaluate(self, params, t):
        amp = params[0]
        r = params[1]
        return amp * tf.exp(-r * t)
        
    def update_initial_posterior(self, t, data, post):
        post[self.param_idx("amp1")] = tf.reduce_max(data)
        post[self.param_idx("r1")] = 0.5

class BiExpModel(Model):

    def __init__(self, options):
        Model.__init__(self, options)
        self.params.append(Parameter("amp1", prior_mean=1, prior_var=10))
        self.params.append(Parameter("amp2", prior_mean=1, prior_var=10))
        self.params.append(Parameter("r1", prior_mean=1, prior_var=10))
        self.params.append(Parameter("r2", prior_mean=1, prior_var=10))
    
    def evaluate(self, params, t):
        amp1 = params[0]
        amp2 = params[1]
        r1 = params[2]
        r2 = params[3]
        return amp1 * tf.exp(-r1 * t) + amp2 * tf.exp(-r2 * t)

    def update_initial_posterior(self, t, data, post):
        post[self.param_idx("amp1")] = 0.9*tf.reduce_max(data)
        post[self.param_idx("amp2")] = 0.1*tf.reduce_max(data)
        post[self.param_idx("r1")] = 0.5
        post[self.param_idx("r2")] = 0.1
        