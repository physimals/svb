import tensorflow as tf
import numpy as np

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
        
        :param t: Sequence of time values of length N
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is MxN tensor where M is the number of voxels. This
                      may be supplied as a PxMxN tensor where P is the number of 
                      parameters.
                      
        :return: MxN tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        raise NotImplementedError("evaluate")

    def ievaluate(self, params, t):
        """
        Evaluate the model outside of a TensorFlow session

        Same as :func:`evaluate` but will run the evaluation
        within a session and return the evaluated output tensor
        """
        with tf.Session():
            return self.evaluate(params, t).eval()

    def update_initial_posterior(self, t, data, post):
        """
        Optional method to update the initial posterior mean values

        :param t: Tensor of time values of length N
        :param data: MxN tensor containing input data, where M is the number of voxels
        :param post: Sequence of tensors of length M, one for each model parameter. This
                     sequence can be updated by indexing to initialize the posterior
                     distribution mean for the corresponding parameter at each voxel
        """
        pass

    def test_data(self, t, params_map):
        """
        Generate test data by evaluating the model on known parameter values
        with optional added noise
        
        :param t: Sequence of time values of length N
        :param params_map: Mapping from parameter name either a single parameter
                           value or a sequence of M parameter values. The special
                           key ``noise_sd``, if present, should containing the 
                           standard deviation of Gaussian noise to add to the 
                           output.
        :return If noise is present, a tuple of two MxN Numpy arrays. The first
                contains the 'clean' output data without noise, the second 
                contains the noisy data. If noise is not present, only a single
                array is returned. 
        """
        param_values = None
        for idx, param in enumerate(self.params):
            if param.name not in params_map:
                raise IndexError("Required parameter not found: %s" % param.name)
            elif isinstance(params_map[param.name], (float, int)):
                value_sequence = np.reshape([params_map[param.name]], (1, 1))
            else:
                # FIXME check if sequence type
                value_sequence = np.reshape(params_map[param.name], (-1, 1))

            if param_values is None:
                param_values = np.zeros((len(self.params), len(value_sequence), len(t)))
            
            if len(value_sequence) != param_values.shape[1]:
                raise ValueError("Parameter %s has wrong number of values: %i (expected %i)" % (param.name, len(value_sequence), param_values.shape[1]))
            else:
                param_values[idx, :, :] = value_sequence

        with tf.Session():
            clean = self.evaluate(param_values, t).eval()
            if "noise_sd" in params_map:
                np.random.seed(1)
                noisy = np.random.normal(clean, params_map["noise_sd"])
                return clean, noisy
            else:
                return clean

class ExpModel(Model):
    """
    Simple exponential decay model
    """
    
    def __init__(self, options):
        Model.__init__(self, options)
        self.params.append(Parameter("amp1", prior_mean=0, prior_var=100))
        self.params.append(Parameter("r1", prior_mean=0, prior_var=100))
    
    def evaluate(self, params, t):
        amp = params[0]
        r = params[1]
        return amp * tf.exp(-r * t)
        
    def update_initial_posterior(self, t, data, post):
        post[self.param_idx("amp1")] = tf.reduce_max(data, axis=1)
        post[self.param_idx("r1")] = tf.fill([tf.shape(data)[0]], 0.5)
        
class BiExpModel(Model):
    """
    Exponential decay with two independent decay rates
    """

    def __init__(self, options):
        Model.__init__(self, options)
        self.params.append(Parameter("amp1", prior_mean=0, prior_var=100))
        self.params.append(Parameter("amp2", prior_mean=0, prior_var=100))
        self.params.append(Parameter("r1", prior_mean=0, prior_var=100))
        self.params.append(Parameter("r2", prior_mean=0, prior_var=100))
    
    def evaluate(self, params, t):
        amp1 = params[0]
        amp2 = params[1]
        r1 = params[2]
        r2 = params[3]
        return amp1 * tf.exp(-r1 * t) + amp2 * tf.exp(-r2 * t)

    def update_initial_posterior(self, t, data, post):
        post[self.param_idx("amp1")] = 0.9*tf.reduce_max(data, axis=1)
        post[self.param_idx("amp2")] = 0.1*tf.reduce_max(data, axis=1)
        post[self.param_idx("r1")] = tf.fill([tf.shape(data)[0]], 0.5)
        post[self.param_idx("r2")] = tf.fill([tf.shape(data)[0]], 0.1)
        