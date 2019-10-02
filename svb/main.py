"""
Implementation of command line tool for SVB

Examples::

    svb --data=asldata.nii.gz --mask=bet_mask.nii.gz
        --model=aslrest --epochs=200 --output=svb_out
"""
import os
import sys
import logging
import logging.config
import argparse
import re

import numpy as np
import nibabel as nib

from . import __version__, SvbFit
from .models import get_model_class
from .data import DataModel

USAGE = "svb <options>"

class SvbArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser for SVB options
    """

    PARAM_OPTIONS = {
        "prior_mean" : float,
        "prior_var" : float,
        "prior_dist" : str,
        "prior_type" : str,
    }

    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="svb", usage=USAGE, **kwargs)

        group = self.add_argument_group("Main Options")
        group.add_argument("--data", dest="data_fname",
                         help="Timeseries input data")
        group.add_argument("--mask", dest="mask_fname",
                         help="Optional voxel mask")
        group.add_argument("--model", dest="model_name",
                         help="Model name")
        group.add_argument("--output",
                         help="Output folder name",
                         default="svb_out")
        group.add_argument("--log-level",
                         help="Logging level - defaults to INFO")
        group.add_argument("--log-config",
                         help="Optional logging configuration file, overrides --log-level")

        group = self.add_argument_group("Inference options")
        group.add_argument("--infer-covar",
                         help="Infer a full covariance matrix",
                         action="store_true", default=False)
        group.add_argument("--force-num-latent-loss",
                         help="Force numerical calculation of the latent loss function",
                         action="store_true", default=False)
        group.add_argument("--allow-nan",
                         dest="suppress_nan",
                         help="Do not suppress NaN values in posterior",
                         action="store_false", default=True)

        group = self.add_argument_group("Training options")
        group.add_argument("--epochs",
                         help="Number of training epochs",
                         type=int, default=100)
        group.add_argument("--learning_rate", "--lr",
                         help="Initial learning rate",
                         type=float, default=0.1)
        group.add_argument("--batch_size", "--bs",
                         help="Batch size. If not specified data will not be processed in batches",
                         type=int)
        group.add_argument("--sample-size", "--ss",
                         help="Sample size for drawing samples from posterior",
                         type=int, default=20)
        group.add_argument("--max-trials",
                         help="Number of epochs without improvement in the cost before reducing the learning rate",
                         type=int, default=50)
        group.add_argument("--lr-quench",
                         help="Quench factor for learning rate when cost does not improve after <conv-trials> epochs",
                         type=float, default=0.99)
        group.add_argument("--lr-min",
                         help="Minimum learning rate",
                         type=float, default=0.00001)

    def parse_args(self, argv=None, namespace=None):
        # Parse built-in fixed options but skip unrecognized options as they may be
        # parameter-specific options.
        options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        # Support arguments of the form --param-<param name>-<param option>
        # (e.g. --param-ftiss-mean=4.4 --param-delttiss-prior-type M)
        param_arg = re.compile("--param-(\w+)-([\w-]+)")
        options.param_overrides = {}
        consume_next_arg = None
        for arg in extras:
            if consume_next_arg:
                if arg.startswith("-"):
                    raise ValueError("Value for parameter option cannot start with - : %s" % arg)
                param, thing = consume_next_arg
                options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](arg)
                consume_next_arg = None
            else:
                kv = arg.split("=", 1)
                key = kv[0]
                match = param_arg.match(key)
                if match:
                    param, thing = match.group(1), match.group(2)

                    # Use underscore for compatibility with kwargs
                    thing = thing.replace("-", "_")
                    if thing not in self.PARAM_OPTIONS:
                        raise ValueError("Unrecognized parameter option: %s" % thing)

                    if param not in options.param_overrides:
                        options.param_overrides[param] = {}
                    if len(kv) == 2:
                        options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](kv[1])
                    else:
                        consume_next_arg = (param, thing)
                else:
                    raise ValueError("Unrecognized argument: %s" % arg)

        return options

def main():
    """
    Command line tool entry point
    """
    try:
        arg_parser = SvbArgumentParser()
        options = arg_parser.parse_args()

        if not options.data_fname:
            raise ValueError("Input data not specified")
        if not options.model_name:
            raise ValueError("Model name not specified")

        welcome = "Welcome to SVB %s" % __version__
        print(welcome)
        print("=" * len(welcome))
        runtime, _ = run(tee=sys.stdout, **vars(options))
        print("FINISHED - runtime %.3fs" % runtime)
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write("ERROR: %s\n" % str(exc))
        import traceback
        traceback.print_exc()

def run(data_fname, model_name, output, mask_fname=None, **kwargs):
    """
    Run model fitting on a data set

    :param data: File name of 4D NIFTI data set containing data to be fitted
    :param model_name: Name of model we are fitting to
    :param output: Output directory, will be created if it does not exist
    :param mask: Optional file name of 3D Nifti data set containing data voxel mask

    All keyword arguments are passed to constructor of the model, the ``SvbFit``
    object and the ``SvbFit.train`` method.
    """
    # Create output directory
    _makedirs(output, exist_ok=True)
    
    _setup_logging(output, **kwargs)
    log = logging.getLogger(__name__)
    log.info("SVB %s", __version__)

    # Initialize the data model which contains data dimensions, number of time
    # points, list of unmasked voxels, etc
    data_model = DataModel(data_fname, mask_fname)
    
    # Create the generative model
    fwd_model = get_model_class(model_name)(**kwargs)
    log.info("Created model: %s", str(fwd_model))

    # Get the time points from the model
    tpts = fwd_model.tpts(data_model)
    if tpts.ndim > 1 and tpts.shape[0] > 1:
        tpts = tpts[data_model.mask_flattened > 0]

    # Train model
    svb = SvbFit(data_model, fwd_model, **kwargs)
    runtime, ret = _runtime(svb.train, tpts, data_model.data_flattened, **kwargs)
    log.info("DONE: %.3fs", runtime)

    # Get output, transposing as required so first index is by parameter
    mean_cost_history = ret[0]
    cost_history_v = ret[2]
    param_history_v = ret[3]
    modelfit = ret[4]
    means = svb.output("model_params")
    variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    _makedirs(output, exist_ok=True)
    for idx, param in enumerate(svb.params):
        data_model.nifti_image(means[idx]).to_filename(os.path.join(output, "mean_%s.nii.gz" % param.name))
        data_model.nifti_image(variances[idx]).to_filename(os.path.join(output, "var_%s.nii.gz" % param.name))

    # Write out voxelwise cost history
    data_model.nifti_image(cost_history_v).to_filename(os.path.join(output, "cost_history.nii.gz"))

    # Write out voxelwise parameter history
    for idx, param in enumerate(svb.params):
        data_model.nifti_image(param_history_v[:, :, idx]).to_filename(os.path.join(output, "mean_%s_history.nii.gz" % param.name))

    # Write out modelfit
    data_model.nifti_image(modelfit).to_filename(os.path.join(output, "modelfit.nii.gz"))

    # Write out runtime
    with open(os.path.join(output, "runtime"), "w") as runtime_f:
        runtime_f.write("%f\n" % runtime)

    log.info("Output written to: %s", output)
    return runtime, mean_cost_history

def _setup_logging(output, **kwargs):
    """
    Set the log level, formatters and output streams for the logging output

    By default this goes to <outdir>/logfile at level INFO
    """
    # First we clear all loggers from previous runs
    for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ['']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []

    if kwargs.get("log_config", None):
        # User can supply a logging config file which overrides everything else
        logging.config.fileConfig(kwargs["log_config"])
    else:
        # By default we send the log to an output logfile
        logfile = os.path.join(output, "logfile")
        level = kwargs.get("log_level", "info")
        if not level:
            level = "info"
        level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(filename=logfile, filemode="w", level=level)

        if kwargs.get("tee", None) is not None:
            # Can also supply a stream to send log output to as well (e.g. sys.stdout)
            extra_handler = logging.StreamHandler(kwargs["tee"])
            extra_handler.setLevel(level)
            extra_handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
            logging.getLogger('').addHandler(extra_handler)

def _runtime(runnable, *args, **kwargs):
    """
    Record how long it took to run something
    """
    import time
    start_time = time.time()
    ret = runnable(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time), ret

def _makedirs(data_vol, exist_ok=False):
    """
    Make directories, optionally ignoring them if they already exist
    """
    try:
        os.makedirs(data_vol)
    except OSError as exc:
        import errno
        if not exist_ok or exc.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    main()
