"""
General utility functions
"""
import logging

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import numpy as np 

TF_DTYPE = tf.float32
NP_DTYPE = np.float32

def ValueList(value_type):
    """
    Class used with argparse for options which can be given as a comma separated list
    """
    def _call(value):
        return [value_type(v) for v in value.replace(",", " ").split()]
    return _call

class LogBase(object):
    """
    Base class that provides a named log and the ability to log tensors easily
    """
    def __init__(self, **kwargs):
        self.log = logging.getLogger(type(self).__name__)

    def log_tf(self, tensor, level=logging.DEBUG, **kwargs):
        """
        Log a tensor

        :param tensor: tf.Tensor
        :param level: Logging level (default: DEBUG)

        Keyword arguments:

        :param summarize: Number of entries to include (default 100)
        :param force: If True, always log this tensor regardless of log level
        :param shape: If True, precede tensor with its shape
        """
        if self.log.isEnabledFor(level) or kwargs.get("force", False):
            if not isinstance(tensor, tf.Tensor):
                tensor = tf.constant(tensor, dtype=TF_DTYPE)
            items = [tensor]
            if kwargs.get("shape", False):
                items.insert(0, tf.shape(tensor))
            return tf.Print(tensor, items, "\n%s" % kwargs.get("name", tensor.name),
                            summarize=kwargs.get("summarize", 100))
        else:
            return tensor

def scipy_to_tf_sparse(scipy_sparse):
    """Converts a scipy sparse matrix to TF representation"""

    spmat = scipy_sparse.tocoo()
    return tf.SparseTensor(
        indices=np.array([
            spmat.row, spmat.col]).T,
        values=spmat.data.astype(NP_DTYPE), 
        dense_shape=spmat.shape, 
    )