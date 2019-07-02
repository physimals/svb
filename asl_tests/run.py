"""
Run ASL tests
"""
import sys
import os
import logging.config

import nibabel as nib
import numpy as np
import tensorflow as tf

sys.path.append("..")
from svb.models.asl import AslRestModel
from svb.svb import SvbFit

# Test data properties
FNAME_RPT = "mpld_asltc_diff.nii.gz"
FNAME_MEAN = "mpld_asltc_diff_mean.nii.gz"
FNAME_MASK = "mpld_asltc_mask.nii.gz"
PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SLICEDT = 0.0452
TAU = 1.8
RPTS = 8

# Optimization properties
EPOCHS = 500

def test_asl(fname, mask=None, outdir=".", **kwargs):
    """
    Fit to a 4D ASL data

    :param fname: File name of Nifti image
    """
    # Create model
    model = AslRestModel(debug=False, **kwargs)

    # Load data and mask
    nii = nib.load(fname)
    fulldata = nii.get_data()
    fullshape = list(fulldata.shape)
    shape = fulldata.shape[:3]
    nt = fulldata.shape[3]

    if mask is not None:
        nii_mask = nib.load(mask)
        d_mask = nii_mask.get_data()
        flatdata = fulldata[d_mask > 0]
    else:
        flatdata = fulldata.reshape(-1, nt)

    # Generate timeseries
    # FIXME derivable from the model?
    # FIXME assuming grouped by PLDs
    plds, repeats = kwargs["plds"], kwargs["repeats"]
    slicedt = kwargs.get("slicedt", 0)
    if slicedt > 0:
        # Generate voxelwise timings array using the slicedt value
        t = np.zeros(fullshape)
        for z in range(fullshape[2]):
            t[:, :, z, :] = np.array(sum([[pld + model.tau + z*slicedt] * repeats for pld in plds], []))
        if mask is not None:
            t = t[d_mask > 0]
        else:
            t = t.reshape(-1, nt)
    else:
        # Timings are the same for all voxels
        t = np.array(sum([[pld + model.tau] * repeats for pld in plds], [])).reshape(1, -1)

    # Example evaluation (ftiss=10, delt=0.7)
    ftiss, delt = 10.0, 0.7
    params = np.zeros((2, 1, nt), dtype=np.float32)
    params[0, :, :] = ftiss
    params[1, :, :] = delt
    print("Sample model output (ftiss=%f, delt=%f): %s" % (ftiss, delt, model.ievaluate(params, t[0])))

    # Do brief initial train with no correlation between parameters, then use this to do full correlation
    # training. The latter can be unstable if not initialized like this.
    svb = SvbFit(model, **kwargs)
    ret = svb.train(t, flatdata, **kwargs)

    # Create output directory
    makedirs(outdir, exist_ok=True)

    mean_cost_history = ret[0]

    # Transpose output so the first index is by parameter
    means = svb.output("model_params")
    variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    makedirs(outdir, exist_ok=True)
    for idx, param in enumerate(model.params):
        nii_mean = _nifti_image(means[idx], shape, d_mask, ref_nii=nii)
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = _nifti_image(variances[idx], shape, d_mask, ref_nii=nii)
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out voxelwise cost history
    cost_history_v = ret[2]
    cost_history_v_nii = _nifti_image(ret[2], shape, d_mask, ref_nii=nii, nt=cost_history_v.shape[1])
    cost_history_v_nii.to_filename(os.path.join(outdir, "cost_history.nii.gz"))

    # Write out voxelwise parameter history
    param_history_v = ret[3]
    for idx, param in enumerate(model.params):
        nii_mean = _nifti_image(param_history_v[:, :, idx], shape, d_mask, ref_nii=nii, nt=cost_history_v.shape[1])
        nii_mean.to_filename(os.path.join(outdir, "mean_%s_history.nii.gz" % param.name))

    # Noise history
    nii_mean = _nifti_image(param_history_v[:, :, model.nparams], shape, d_mask, ref_nii=nii, nt=cost_history_v.shape[1])
    nii_mean.to_filename(os.path.join(outdir, "mean_noise_history.nii.gz"))

    # Write out modelfit
    fit_nii = _nifti_image(ret[4], shape, d_mask, ref_nii=nii, nt=nt)
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

    return mean_cost_history

def _nifti_image(data, shape, mask, ref_nii, nt=1):
    if nt > 1:
        shape = list(shape) + [nt]
    ndata = np.zeros(shape, dtype=np.float)
    ndata[mask > 0] = data
    return nib.Nifti1Image(ndata, None, header=ref_nii.header)

def runtime(runnable, *args, **kwargs):
    """
    Record how long it took to run something
    """
    import time
    t0 = time.time()
    ret = runnable(*args, **kwargs)
    t1 = time.time()
    return (t1-t0), ret

def makedirs(d, exist_ok=False):
    """
    Make directories, optionally ignoring them if they already exist
    """
    try:
        os.makedirs(d)
    except OSError as e:
        import errno
        if not exist_ok or e.errno != errno.EEXIST:
            raise

def mean_learning_rate():
    """
    Run test on mean across repeats varying learning rate
    """
    learning_rates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)

    for lr in learning_rates:
        mean_cost_history = test_asl(
            fname=FNAME_MEAN,
            mask=FNAME_MASK,
            plds=PLDS,
            slicedt=SLICEDT,
            repeats=1,
            tau=TAU,
            outdir="mean_lr_%.3f" % lr,
            training_epochs=EPOCHS,
            batch_size=len(PLDS),
            learning_rate=lr,
            quench_rate=0.95)

        print(lr, mean_cost_history[-1])

def rpts_learning_rate():
    """
    Run test on full repeated data varying learning rate and batch size
    """
    learning_rates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (5, 6, 10, 12, 20, 24)

    for bs in batch_sizes:
        for lr in learning_rates:
            mean_cost_history = test_asl(
                fname=FNAME_RPT,
                mask=FNAME_MASK,
                plds=PLDS,
                slicedt=SLICEDT,
                repeats=RPTS,
                tau=TAU,
                outdir="rpts_lr_%.3f_bs_%i" % (lr, bs),
                training_epochs=EPOCHS,
                batch_size=bs,
                learning_rate=lr,
                quench_rate=0.95)

            print(lr, bs, mean_cost_history[-1])

def _run():
    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    if os.path.exists("logging.conf"):
        logging.config.fileConfig("logging.conf")

    #mean_learning_rate()
    rpts_learning_rate()

if __name__ == "__main__":
    _run()
