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
from optparse import OptionGroup, OptionParser

import numpy as np
import nibabel as nib

from . import __version__, SvbFit
from .models import get_model_class

USAGE = "svb <options>"

class SvbOptionParser(OptionParser):
    """
    OptionParser for SVB options
    """
    def __init__(self, **kwargs):
        OptionParser.__init__(self, usage=USAGE, version=__version__, **kwargs)

        group = OptionGroup(self, "Main Options")
        group.add_option("--data",
                         help="Timeseries input data")
        group.add_option("--mask",
                         help="Optional voxel mask")
        group.add_option("--model",
                         help="Model name")
        group.add_option("--output",
                         help="Output folder name",
                         default="svb_out")
        group.add_option("--log-level",
                         help="Logging level - defaults to INFO")
        group.add_option("--log-config",
                         help="Optional logging configuration file, overrides --log-level")
        self.add_option_group(group)

        group = OptionGroup(self, "Inference options")
        group.add_option("--infer-covar",
                         help="Infer a full covariance matrix",
                         action="store_true", default=False)
        group.add_option("--force-num-latent-loss",
                         help="Force numerical calculation of the latent loss function",
                         action="store_true", default=False)
        self.add_option_group(group)

        group = OptionGroup(self, "Training options")
        group.add_option("--epochs",
                         help="Number of training epochs",
                         type="int", default=100)
        group.add_option("--learning_rate", "--lr",
                         help="Initial learning rate",
                         type="float", default=0.1)
        group.add_option("--batch_size", "--bs",
                         help="Batch size. If not specified data will not be processed in batches",
                         type="int")
        group.add_option("--sample-size", "--ss",
                         help="Sample size for drawing samples from posterior",
                         type="int", default=20)
        group.add_option("--max-trials",
                         help="Number of epochs without improvement in the cost before reducing the learning rate",
                         type="int", default=50)
        group.add_option("--lr-quench",
                         help="Quench factor for learning rate when cost does not improve after <conv-trials> epochs",
                         type="float", default=0.99)
        group.add_option("--lr-min",
                         help="Minimum learning rate",
                         type="float", default=0.00001)
        self.add_option_group(group)

def main():
    """
    Command line tool entry point
    """
    try:
        opt_parser = SvbOptionParser()
        options, _ = opt_parser.parse_args()

        if not options.data:
            raise ValueError("Input data not specified")
        if not options.model:
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

def calc_neighbours(mask_vol):
    """
    Generate nearest neighbour and second nearest neighbour lists
    """
    # First nearest neighbour lists.
    # Note that Numpy uses (by default) C-style row-major ordering
    # for voxel indices so the index is z + y*nz + x*ny*nz
    # Also we need to check that potential neighbours are not masked
    def add_if_unmasked(x, y, z, masked_indices, nns):
        idx  = masked_indices[x, y, z]
        if idx >= 0:
            nns.append(idx)

    shape = mask_vol.shape
    masked_indices = np.full(shape, -1, dtype=np.int)
    nx, ny, nz = tuple(shape)
    voxel_idx = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask_vol[x, y, z] > 0:
                    masked_indices[x, y, z] = voxel_idx
                    voxel_idx += 1

    voxel_nns = []
    indices_nn = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask_vol[x, y, z] > 0:
                    nns = []
                    if x > 0: add_if_unmasked(x-1, y, z, masked_indices, nns)
                    if x < nx-1: add_if_unmasked(x+1, y, z, masked_indices, nns)
                    if y > 0: add_if_unmasked(x, y-1, z, masked_indices, nns)
                    if y < ny-1: add_if_unmasked(x, y+1, z, masked_indices, nns)
                    if z > 0: add_if_unmasked(x, y, z-1, masked_indices, nns)
                    if z < nz-1: add_if_unmasked(x, y, z+1, masked_indices, nns)
                    voxel_nns.append(nns)
                    # For TensorFlow sparse tensor
                    voxel_idx = z + y*nz + x*ny*nz
                    for nn in nns:
                        indices_nn.append([voxel_idx, nn])

    # Second nearest neighbour lists exclude self but include duplicates
    voxel_n2s = [[] for voxel in voxel_nns]
    indices_n2 = []
    for voxel_idx, nns in enumerate(voxel_nns):
        for nn in nns:
            voxel_n2s[voxel_idx].extend(voxel_nns[nn])
        voxel_n2s[voxel_idx] = [v for v in voxel_n2s[voxel_idx] if v != voxel_idx]
        for n2 in voxel_n2s[voxel_idx]:
            indices_n2.append([voxel_idx, n2])
    
    return indices_nn, indices_n2

def run(data, model, output, mask=None, **kwargs):
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

    # Load the input data
    nii = nib.load(data)
    data_vol = nii.get_data()
    shape = list(data_vol.shape)[:3]
    n_tpts = data_vol.shape[3]
    data_flattened = data_vol.reshape(-1, n_tpts)
    log.info("Loaded data from %s", data)

    # Create the generative model
    model = get_model_class(model)(**kwargs)
    log.info("Created model: %s", str(model))

    # Get the time points from the model
    tpts = model.tpts(n_tpts=n_tpts, shape=shape)

    # If there is a mask load it and use it to mask the data
    if mask:
        mask_nii = nib.load(mask)
        mask_vol = mask_nii.get_data()
        mask_flattened = mask_vol.flatten()
        data_flattened = data_flattened[mask_flattened > 0]
        if tpts.ndim > 1 and tpts.shape[0] > 1:
            tpts = tpts[mask_flattened > 0]
        log.info("Loaded mask from %s", data)
    else:
        mask_vol = np.ones(shape)

    indices_nn, indices_n2 = calc_neighbours(mask_vol)

    # Train model
    svb = SvbFit(model, indices_nn=indices_nn, indices_n2=indices_n2, n_unmasked_voxels=data_flattened.shape[0], **kwargs)
    log.info("Training model...")
    runtime, ret = _runtime(svb.train, tpts, data_flattened, **kwargs)
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
    for idx, param in enumerate(model.params):
        nii_mean = _nifti_image(means[idx], shape, mask_vol, ref_nii=nii)
        nii_mean.to_filename(os.path.join(output, "mean_%s.nii.gz" % param.name))
        nii_var = _nifti_image(variances[idx], shape, mask_vol, ref_nii=nii)
        nii_var.to_filename(os.path.join(output, "var_%s.nii.gz" % param.name))

    # Write out voxelwise cost history
    cost_history_v_nii = _nifti_image(cost_history_v, shape, mask_vol,
                                      ref_nii=nii, n_tpts=cost_history_v.shape[1])
    cost_history_v_nii.to_filename(os.path.join(output, "cost_history.nii.gz"))

    # Write out voxelwise parameter history
    for idx, param in enumerate(model.params):
        nii_mean = _nifti_image(param_history_v[:, :, idx], shape, mask_vol,
                                ref_nii=nii, n_tpts=cost_history_v.shape[1])
        nii_mean.to_filename(os.path.join(output, "mean_%s_history.nii.gz" % param.name))

    # Noise history
    nii_mean = _nifti_image(param_history_v[:, :, model.nparams], shape, mask_vol,
                            ref_nii=nii, n_tpts=cost_history_v.shape[1])
    nii_mean.to_filename(os.path.join(output, "mean_noise_history.nii.gz"))

    # Write out modelfit
    fit_nii = _nifti_image(modelfit, shape, mask_vol, ref_nii=nii, n_tpts=n_tpts)
    fit_nii.to_filename(os.path.join(output, "modelfit.nii.gz"))

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

def _nifti_image(data, shape, mask, ref_nii, n_tpts=1):
    """
    :return: A nibabel.Nifti1Image for some, potentially masked, output data
    """
    if n_tpts > 1:
        shape = list(shape) + [n_tpts]
    ndata = np.zeros(shape, dtype=np.float)
    ndata[mask > 0] = data
    return nib.Nifti1Image(ndata, None, header=ref_nii.header)

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
