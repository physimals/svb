"""
SVB - Model parameters

This module defines a set of classes of model parameters.

The factory methods which create priors/posteriors can
make use of the instance class to create the appropriate
type of vertexwise prior/posterior
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
   
from svb.posterior import GaussianGlobalPosterior
from .utils import LogBase
from . import dist

def get_parameter(name, **kwargs):
    """
    Factory method to create an instance of a parameter
    """
    custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
    kwargs.update(custom_kwargs)

    # FIXME: hack because when var = 1 the LogNormal gives inf latent cost during training
    if (kwargs.get('dist') == 'LogNormal') and (kwargs.get('var') == 1.0):
        raise RuntimeError('LogNormal distribution cannot have initial var = 1.0') 

    data_space = kwargs.get("data_space", "voxel")
    desc = kwargs.get("desc", "No description given")
    prior_dist = dist.get_dist(prefix="prior", **kwargs)
    prior_type = kwargs.get("prior_type", "N")
    post_dist = dist.get_dist(prefix="post", **kwargs)
    post_init = kwargs.get("post_init", None)

    return Parameter(name, desc=desc, prior=prior_dist, prior_type=prior_type,
        post=post_dist, post_init=post_init, data_space=data_space)

class Parameter(LogBase):
    """
    A standard model parameter
    """

    def __init__(self, name, **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param prior: Dist instance giving the parameter's prior distribution
        :param desc: Optional parameter description

        Keyword arguments (optional):
         - ``mean_init`` Initial value for the posterior mean either as a numeric
                         value or a callable which takes the parameters t, data, param_name
         - ``log_var_init`` Initial value for the posterior log variance either as a numeric
                            value or a callable which takes the parameters t, data, param_name
         - ``data_space`` Space in which parameter is defined/estimated, either 
                          "voxel" (default) or "node" for volume/surface respectively. 
         - ``param_overrides`` Dictionary keyed by parameter name. Value should be dictionary
                               of keyword arguments which will override those defined as
                               existing keyword arguments
        """
        LogBase.__init__(self)

        custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_kwargs)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.prior_dist = kwargs.get("prior")
        self.prior_type = kwargs.get("prior_type", "N")
        self.post_dist = kwargs.get("post", self.prior_dist)
        self.post_init = kwargs.get("post_init", None)  
        self.data_space = kwargs.get("data_space", "voxel")


    def __str__(self):
        return "Parameter: %s" % self.name

    def __repr__(self):
        return ("Parameter: %s, %s space, %s prior type" 
            % (self.name, self.data_space, self.prior_type))

    @property
    def is_global(self):
        """
        If global, parameter has a single global value across all 
        voxels or nodes (if in volume/surface space respectively). 
        """
        return isinstance(self, GaussianGlobalPosterior)
