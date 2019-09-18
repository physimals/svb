"""
Base class for a forward model whose parameters are to be fitted
"""
import tensorflow as tf
import numpy as np

from .utils import LogBase

class Model(LogBase):
    """
    A forward model

    :attr params: Sequence of ``Parameter`` objects
    :attr nparams: Number of model parameters
    """

    def __init__(self, **options):
        LogBase.__init__(self)
        self.params = []
        self._t0 = options.get("t0", 0)
        self._dt = options.get("dt", 1)

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

    def tpts(self, data_model):
        """
        Get the full set of timeseries time values

        :param n_tpts: Number of time points required for the data to be fitted
        :param shape: Shape of source data which may affect the times assigned

        By default this is a linear space using the attributes ``t0`` and ``dt``.
        Some models may have time values fixed by some other configuration. If
        the number of time points is fixed by the model it must match the
        supplied value ``n_tpts``.

        :return: Either a Numpy array of shape [n_tpts] or a Numpy array of shape
                 shape + [n_tpts] for voxelwise timepoints
        """
        return np.linspace(self._t0, self._t0+data_model.n_tpts*self._dt, num=data_model.n_tpts, endpoint=False)

    def evaluate(self, params, tpts):
        """
        Evaluate the model

        :param t: Time values to evaluate the model at, supplied as a tensor of shape
                  [1x1xB] (if time values at each voxel are identical) or [Vx1xB]
                  otherwise.
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is VxSx1 tensor where V is the number of voxels and
                      S is the number of samples per parameter. This
                      may be supplied as a PxVxSx1 tensor where P is the number of
                      parameters.

        :return: [VxSxB] tensor containing model output at the specified time values
                 for each voxel, and each sample (set of parameter values).
        """
        raise NotImplementedError("evaluate")

    def ievaluate(self, params, tpts):
        """
        Evaluate the model outside of a TensorFlow session

        Same as :func:`evaluate` but will run the evaluation
        within a session and return the evaluated output tensor
        """
        with tf.Session():
            return self.evaluate(tf.constant(params, dtype=tf.float32), tf.constant(tpts, dtype=tf.float32)).eval()

    def test_data(self, tpts, params_map):
        """
        Generate test data by evaluating the model on known parameter values
        with optional added noise

        :param tpts: 1xN or MxN tensor of time values (possibly varying by voxel)
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
                param_values = np.zeros((len(self.params), len(value_sequence), len(tpts)))

            if len(value_sequence) != param_values.shape[1]:
                raise ValueError("Parameter %s has wrong number of values: %i (expected %i)" %
                                 (param.name, len(value_sequence), param_values.shape[1]))
            else:
                param_values[idx, :, :] = value_sequence

        with tf.Session():
            clean = self.evaluate(param_values, tpts).eval()
            if "noise_sd" in params_map:
                np.random.seed(1)
                noisy = np.random.normal(clean, params_map["noise_sd"])
                return clean, noisy
            else:
                return clean
