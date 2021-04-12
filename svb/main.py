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
from functools import partial

import numpy as np
import nibabel as nib

from . import __version__, SvbFit, get_model_class
from .model import assert_param_overrides_used
from .utils import ValueList
from .data import SurfaceModel, VolumetricModel, HybridModel

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
        "post_mean" : float,
        # FIXME: allow user to set global: True as option here 
    }

    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="svb", usage=USAGE, add_help=False, **kwargs)

        group = self.add_argument_group("Main Options")
        group.add_argument("--data",
                         help="Timeseries input data")
        group.add_argument("--mask",
                         help="Optional voxel mask")
        group.add_argument("--mode", 
                         help="""Inference domain. For 'surface' or 'hybrid', see further options below.""", 
                         choices=['volume', 'surface', 'hybrid'], default='volume')
        group.add_argument("--post-init", dest="initial_posterior",
                         help="Initialize posterior from data file saved using --output-post")
        group.add_argument("--model", dest="model_name",
                         help="Model name")
        group.add_argument("--output",
                         help="Output folder name",
                         default="svb_out")
        group.add_argument("--log-level",
                         help="Logging level - defaults to INFO")
        group.add_argument("--log-config",
                         help="Optional logging configuration file, overrides --log-level")
        group.add_argument("--help", action="store_true", default=False,
                         help="Display help")

        group = self.add_argument_group("Surface or hybrid inference options (note that --projector overrides all of the following)")
        group.add_argument("--projector", help="""Path to a toblerone projector file 
            (see 'toblerone -prepare-projector'). This replaces all below options.""")
        group.add_argument("--fsdir", help="""Required if no --projector specified. Path to FreeSurfer
            subject directory, from which all cortical surfaces will be loaded""")
        group.add_argument("--LPS", help="Alternative to --fsdir, path to left pial surface")
        group.add_argument("--LWS", help="Alternative to --fsdir, path to left white surface")
        group.add_argument("--RPS", help="Alternative to --fsdir, path to right pial surface")
        group.add_argument("--RWS", help="Alternative to --fsdir, path to right white surface")
        group.add_argument("--struct2ref", 
            help="""Required if no --projector specified. Registration transform to align surfaces with the --data volume. Must be in world or FSL FLIRT convention (if FLIRT, also set --flirt).""")
        group.add_argument("--flirt", help="""Set if --struct2ref is a FSL FLIRT
            transform (not in world convention). Also set --struct.""", action='store_true')
        group.add_argument("--struct", help="""Path to source image of --struct2ref
            transform, ie image surfaces are currently aligned to.""")
        
        group = self.add_argument_group("Inference options")
        group.add_argument("--no-covar", 
                         dest="infer_covar",
                         help="Do not infer a full covariance matrix",
                         action="store_false", default=True)
        group.add_argument("--force-num-latent-loss",
                         help="Force numerical calculation of the latent loss function",
                         action="store_true", default=False)
        group.add_argument("--allow-nan",
                         dest="suppress_nan",
                         help="Do not suppress NaN values in posterior",
                         action="store_false", default=True)      
        group.add_argument("--pvcorr", 
                         help="Perform PVEc (volumetric mode only, this is automatic in surface or hybrid mode)", action='store_true', default=False)
        group.add_argument("--pvgm", 
                         help="GM PV estimates for PVEc")     
        group.add_argument("--pvwm", 
                         help="GM PV estimates for PVEc")     

        group = self.add_argument_group("Training options")
        group.add_argument("--epochs",
                         help="Number of training epochs",
                         type=int, default=100)
        group.add_argument("--learning-rate", "--lr",
                         help="Initial learning rate",
                         type=float, default=0.1)
        group.add_argument("--batch-size", "--bs",
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

        group = self.add_argument_group("Output options")
        group.add_argument("--save-input",
                         help="Save input data",
                         action="store_true", default=False)
        group.add_argument("--save-var",
                         help="Save parameter variance",
                         action="store_true", default=False)
        group.add_argument("--save-std",
                         help="Save parameter standard deviation",
                         action="store_true", default=False)
        group.add_argument("--save-param-history",
                         help="Save parameter history by epoch",
                         action="store_true", default=False)
        group.add_argument("--save-noise",
                         help="Save noise parameter (in 'nii' format only)",
                         action="store_true", default=False)
        group.add_argument("--save-cost",
                         help="Save cost",
                         action="store_true", default=False)
        group.add_argument("--save-cost-history",
                         help="Save cost history by epoch",
                         action="store_true", default=False)
        group.add_argument("--save-model-fit",
                         help="Save model fit",
                         action="store_true", default=False)
        group.add_argument("--save-post", "--save-posterior",
                         help="Save full posterior distribution",
                         action="store_true", default=False)
        # FIXME: this is chunky... make sense?
        group.add_argument("--out-format", choices=['nii', 'gii', 'cii', 'flatnii'], 
            nargs='+', help="""Output format for model parameters except noise (multiple 
            may be set). The default (and only choice) for volume mode is 'nii' (NIFTI). 
            The default for surface mode is 'gii' (GIFTI). 
            The default for hybrid mode is 'nii gii' (subcortex and cortex
            respectively). 'cii' (CIFTI) may only be set in hybrid mode. 
            'flatnii' may be set in surface or hybrid mode and projects surface 
            data back into voxel space (and in the case of hybrid mode, merges it
            with existing volume data). 
            """)

    def parse_args(self, argv=None, namespace=None):
        # Parse built-in fixed options but skip unrecognized options as they may be
        #  model-specific option or parameter-specific optionss.
        options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)
                
        # Now we should know the model, so we can add it's options and parse again
        if options.model_name:
            group = self.add_argument_group("%s model options" % options.model_name.upper())
            for model_option in get_model_class(options.model_name).OPTIONS:
                kwargs = {
                    "help" : model_option.desc,
                    "type" : model_option.type,
                    "default" : model_option.default,
                }
                if model_option.units:
                    kwargs["help"] += " (%s)" % model_option.units
                if model_option.default is not None:
                    kwargs["help"] += " - default %s" % str(model_option.default)
                else:
                    kwargs["help"] += " - no default"

                if model_option.type == bool:
                    kwargs["action"] = "store_true"
                    kwargs.pop("type")
                group.add_argument(*model_option.clargs, **kwargs)
            options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        if options.help:
            self.print_help()
            sys.exit(0)

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

        if not options.data:
            raise ValueError("Input data not specified")
        if not options.model_name:
            raise ValueError("Model name not specified")

        # Fixed for CL tool
        options.save_mean = True
        options.save_runtime = True
        options.save_log = True

        welcome = "Welcome to SVB %s" % __version__
        print(welcome)
        print("=" * len(welcome))
        runtime, _, _ = run(log_stream=sys.stdout, **vars(options))
        print("FINISHED - runtime %.3fs" % runtime)
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write("ERROR: %s\n" % str(exc))
        import traceback
        traceback.print_exc()

def run(data, model_name, output, mask=None, **kwargs):
    """
    Run model fitting on a data set

    :param data: File name of 4D NIFTI data set containing data to be fitted
    :param model_name: Name of model we are fitting to
    :param output: output directory, will be created if it does not exist
    :param mask: Optional file name of 3D Nifti data set containing data voxel mask

    All keyword arguments are passed to constructor of the model, the ``SvbFit``
    object and the ``SvbFit.train`` method.
    """
    # Create output directory
    _makedirs(output, exist_ok=True)
    
    setup_logging(output, **kwargs)
    log = logging.getLogger(__name__)
    log.info("SVB %s", __version__)

    # Set defaults for mode and outformat if not supplied. 
    # If no outformat, set as None, and if there is one, 
    # make sure it is a list 
    mode = kwargs.get('mode', 'volume')
    outformat = kwargs.get('outformat', None)
    if outformat and not isinstance(outformat, list):
        outformat = [outformat]

    # Initialize the data model which contains data dimensions, number of time
    # points, list of unmasked voxels, etc
    if mode == 'volume': 
        if outformat is None: 
            outformat = ['nii']
        elif outformat != ['nii']:
            raise ValueError("outformat must be 'nii' in volume mode")
        data_model = VolumetricModel(data, mask=mask, **kwargs)
    elif mode == 'surface':
        if outformat is None: 
            outformat = ['gii']
        elif (set(outformat) & set(['cii', 'nii'])): 
            illegal = list(set(outformat) & set(['cii', 'nii']))
            raise ValueError("outformat cannot be {} in surface mode"
                                .format(illegal))
        data_model = SurfaceModel(data, mask=mask, **kwargs)
    elif mode =='hybrid': 
        if outformat is None: 
            outformat = ['nii', 'gii']
        data_model = HybridModel(data, mask=mask, **kwargs)
    else: 
        raise ValueError("mode must be 'volume', 'surface' or 'hybrid'")
    
    # Set the default "data_space" for parameters. This is set by 
    # the data_model, "voxel" means voxelwise inference, "node" means
    # surface inference. Noise is ALWAYS defined in "voxel" however. 
    if "data_space" not in kwargs:
        data_space = "voxel" if data_model.is_volumetric else "node"
    else:
        data_space = kwargs.pop("data_space")

    # Create the generative model
    fwd_model = get_model_class(model_name)
    fwd_model = fwd_model(data_model, data_space=data_space, **kwargs)
    fwd_model.log_config()

    # Check that any parameter overrides actually match parameters in the model
    assert_param_overrides_used(fwd_model.params, kwargs)

    # Get the time points from the model.
    tpts = fwd_model.tpts()

    # Train model
    svb = SvbFit(data_model, fwd_model, **kwargs)
    runtime, training_history = _runtime(svb.train, tpts, data_model.data_flattened, **kwargs)
    log.info("DONE: %.3fs", runtime)

    _makedirs(output, exist_ok=True)
    if kwargs.get("save_noise", False):
        params = svb.params
    else:
        params = fwd_model.params

    # Prepare to write out results
    means = svb.evaluate(svb.model_means)
    variances = svb.evaluate(svb.model_vars)
    noise = svb.evaluate(svb.noise_mean)

    # Get the formats for output (can be multiple formats), sanity check. 

    if 'gii' in outformat: assert (data_model.is_hybrid or data_model.is_pure_surface)
    if 'cii' in outformat: assert (data_model.is_hybrid or data_model.is_pure_surface)
    if 'nii' in outformat: assert not (data_model.is_pure_surface)
    if 'flatnii' in outformat: assert not (data_model.is_volumetric)

    # In hybrid mode, the inference nodes representing volume and surface are 
    # concatenated together (the data model takes care of this). For extracting 
    # the surface and volume components out, the data model provides slice objects.
    vslice, sslice = slice(0), slice(0) 
    if hasattr(data_model, 'vol_slicer'): 
        vslice = data_model.vol_slicer
    if hasattr(data_model, 'surf_slicer'):
        sslice = data_model.surf_slicer

    # Helper functions for constructing output paths in terms of parameter 
    # name and whether data is volumetric or surface. The partial function
    # gifti_write is pre-configured to split the surface data across each 
    # hemisphere of the cortex, if two are present. 
    makepathbase = lambda s: os.path.join(output, s)
    makevpath = lambda s: makepathbase(f"{s}.nii.gz")
    makespath = lambda s,side: makepathbase(f"{s}_{side}_cortex.func.gii")
    gifti_writer = partial(_write_giftis, 
                            data_model=data_model, path_gen=makespath)

    # Output model parameters 
    for idx, param in enumerate(params):

        # Mean parameter values 
        if kwargs.get("save_mean", True):
            mean = means[idx]
            name = f"mean_{param.name}"
            if 'nii' in outformat: 
                p = makevpath(name)
                data_model.nifti_image(mean[vslice]).to_filename(p)
            if 'gii' in outformat:
                gifti_writer(mean[sslice], name)

        # Variances   
        if kwargs.get("save_var", False):
            var = variances[idx]
            name = f"var_{param.name}"
            if 'nii' in outformat: 
                p = makevpath(name)
                data_model.nifti_image(var[vslice]).to_filename(p)
            if 'gii' in outformat:
                sdata = variances[idx][sslice]
                gifti_writer(sdata, name)

        # Std deviations
        if kwargs.get("save_std", False):
            std = np.sqrt(variances[idx])
            name = f"std_{param.name}"
            if 'nii' in outformat: 
                p = makevpath(name)
                data_model.nifti_image(std[vslice]).to_filename(p)
            if 'gii' in outformat:
                gifti_writer(std[sslice], name)

    # Noise (volumetric only)
    if kwargs.get("save_noise", False):
        p = makevpath("noise")
        data_model.nifti_image(noise).to_filename(p)

    # Reconstruction cost (volumetric only)
    cost_history_v = training_history["reconstruction_cost"]
    if kwargs.get("save_cost", False):
        data_model.nifti_image(cost_history_v[..., -1]).to_filename(makevpath("cost"))
    if kwargs.get("save_cost_history", False):
        data_model.nifti_image(cost_history_v).to_filename(makevpath("cost_history"))

    # Node-wise parameter history 
    if kwargs.get("save_param_history", False):
        param_history = training_history["node_params"]
        for idx, param in enumerate(params):
            phist = param_history[...,idx]
            name = f"mean_{param.name}_history"
            if 'nii' in outformat: 
                p = makevpath(name)
                data_model.nifti_image(phist[vslice,:]).to_filename(p)
            if 'gii' in outformat:
                gifti_writer(phist[sslice,:], name)

    # Model fit across all timepoints (note this can only be a nii volume)
    if kwargs.get("save_model_fit", False):
        p = makevpath("modelfit")
        data_model.nifti_image(svb.modelfit).to_filename(p)

    # Posterior (means and upper half of covariance matrix)
    # FIXME: surface is written out as a GIFTI series following the same 
    # convention as for volume. 
    if kwargs.get("save_post", False):
        name = "posterior.nii.gz"
        if 'nii' in outformat: 
            p = makevpath(name)
            post_data = data_model.posterior_data(
                svb.evaluate(svb.post.mean)[vslice,:], 
                svb.evaluate(svb.post.cov)[vslice,:])
            data_model.nifti_image(post_data).to_filename(p)
            log.info("Volumetric posterior data shape: %s", post_data.shape)
        if 'gii' in outformat:
            post_data = data_model.posterior_data(
                svb.evaluate(svb.post.mean)[sslice,:], 
                svb.evaluate(svb.post.cov)[sslice,:])
            gifti_writer(post_data, name)
            log.info("Surface posterior data shape: %s", post_data.shape)

    # Runtime (text files)
    # FIXME: added .txt extensions
    if kwargs.get("save_runtime", False):
        with open(os.path.join(output, "runtime.txt"), "w") as runtime_f:
            runtime_f.write("%f\n" % runtime)

        runtime_history = training_history["runtime"]
        with open(os.path.join(output, "runtime_history.txt"), "w") as runtime_f:
            for epoch_time in runtime_history:
                runtime_f.write("%f\n" % epoch_time)

    # Input data (volumetric only)
    if kwargs.get("save_input", False):
        p = makevpath("input_data")
        data_model.nifti_image(data_model.data_flattened).to_filename(p)

    # Projector, if it exists. 
    # FIXME: a nice thing to do for repeat runs? this way the user can re-use it if anything
    # goes wrong, but it would be nice to let them know this is default behaviour. 
    if hasattr(data_model, "projector"):
        data_model.projector.save(os.path.join(output, "projector.h5"))

    log.info("Output written to: %s", output)
    return runtime, svb, training_history

def setup_logging(outdir=".", **kwargs):
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
        # Set log level on the root logger to allow for the possibility of 
        # debug logging on individual loggers
        level = kwargs.get("log_level", "info")
        if not level:
            level = "info"
        level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(level)

        if kwargs.get("save_log", False):
            # Send the log to an output logfile
            logfile = os.path.join(outdir, "logfile")
            logging.basicConfig(filename=logfile, filemode="w", level=level)

        if kwargs.get("log_stream", None) is not None:
            # Can also supply a stream to send log output to as well (e.g. sys.stdout)
            extra_handler = logging.StreamHandler(kwargs["log_stream"])
            extra_handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
            logging.getLogger().addHandler(extra_handler)

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

def _write_giftis(sdata, name_base, data_model, path_gen):
    """
    Helper for writing surface data across one or both 
    hemispheres that are present within a data model. Use 
    a partial function call with data_model and path_gen 
    to make a simple 'writer' function. 

    :param sdata: data array, shape (all surface nodes x N). 
    :param name_base: file base name, eg 'mean_ftiss'
    :param data_model: data model used for SVB run 
    :param path_gen: callable that accepts name_base and side, 
                    both strings, and returns a string path

    """
    for hemi, hslice in zip(data_model.projector.iter_hemis, 
                            data_model.iter_hemi_slicers): 
        d = sdata[hslice]
        p = path_gen(name_base, hemi.side)
        g = data_model.gifti_image(d, hemi.side)
        nib.save(g, p)

 
if __name__ == "__main__":
    main()
