"""
General utility functions
"""
import logging

import tensorflow as tf

class LogBase(object):
    """
    Base class that provides a named log and the ability to log tensors easily
    """
    def __init__(self):
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
            items = [tensor]
            if kwargs.get("shape", False):
                items.insert(0, tf.shape(tensor))
            return tf.Print(tensor, items, "\n%s" % kwargs.get("name", tensor.name),
                            summarize=kwargs.get("summarize", 100))
        else:
            return tensor

def debug(obj, tensor, **kwargs):
    """
    Log a tensor if obj.debug = True

    :param tensor: tf.Tensor

    Keyword arguments:

    :param summarize: Number of entries to include (default 100)
    :param force: If True, always log this tensor regardless of log level
    :param shape: If True, precede tensor with its shape
    """
    if kwargs.get("force", False) or obj.debug:
        items = [tensor]
        if kwargs.get("shape", False):
            items.insert(0, tf.shape(tensor))
        return tf.Print(tensor, items, "\n%s" % kwargs.get("name", tensor.name),
                        summarize=kwargs.get("summarize", 100))
    else:
        return tensor
