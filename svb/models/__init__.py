"""
Base classes and built in models for SVB inference
"""
import pkg_resources

from .asl import *
from .exp import *
from .misc import *

MODELS = {
    "aslrest" : AslRestModel,
    "exp" : ExpModel,
    "biexp" : BiExpModel,
    "constant" : ConstantModel,
    "poly" : PolyModel,
}

_models_loaded = False

def get_model_class(model_name):
    """
    Get a model class by name
    """
    global _models_loaded
    if not _models_loaded:
        for model in pkg_resources.iter_entry_points('svb.models'):
            MODELS[model.name] = model.load()
        _models_loaded = True

    model_class = MODELS.get(model_name, None)
    if model_class is None:
        raise ValueError("No such model: %s" % model_name)

    return model_class
