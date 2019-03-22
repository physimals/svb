import sys
import os

import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from asl_model import AslRestModel
from inference import VaeNormalFit

def test_asl(fname, mask=None, learning_rate=0.02, batch_size=10, epochs=100, outdir=".", limit_voxels=None, **kwargs):
    """
    Fit to a 4D ASL data
    
    :param fname: File name of Nifti image
    """
    # Create model
    model = AslRestModel(debug=False, **kwargs)

    # Load data and mask
    nii = nib.load(fname)
    d = nii.get_data()
    shape = d.shape
    d_flat = d.reshape(-1, shape[-1])

    if mask is not None:
        nii_mask = nib.load(mask)
        d_mask = nii_mask.get_data()
        d_flat = d[d_mask > 0]

    # Generate timeseries 
    # FIXME derivable from the model?
    # FIXME assuming grouped by PLDs
    slicedt = kwargs.get("slicedt", 0)
    if slicedt > 0:
        # Generate voxelwise timings array using the slicedt value
        t = np.zeros(shape)
        for z in range(shape[2]):
            t[:, :, z, :] = np.array(sum([[pld + model.tau + z*slicedt] * model.repeats for pld in model.plds], []))
        if mask is not None:
            t = t[d_mask > 0]
        else:
            t = t.reshape(-1, shape[-1])
    else:
        # Timings are the same for all voxels
        t = np.array(sum([[pld + model.tau] * model.repeats for pld in model.plds], [])).reshape(1, -1)

    if limit_voxels is not None:
        # Run on voxel subset
        d_flat = d_flat[:limit_voxels, :].reshape(-1, shape[3])
        t = t[:limit_voxels, :].reshape(-1, shape[3])

    # Example evaluation (ftiss=10, delt=0.7)
    ftiss, delt = 10.0, 0.7
    params = np.zeros((2, 1, shape[3]), dtype=np.float32)
    params[0, :, :] = ftiss
    params[1, :, :] = delt
    print("Sample model output (ftiss=%f, delt=%f): %s" % (ftiss, delt, model.ievaluate(params, t[0])))

    # Do brief initial train with no correlation between parameters, then use this to do full correlation
    # training. The latter can be unstable if not initialized like this.
    vae_init = VaeNormalFit(model, mode_corr='no_post_corr', learning_rate=learning_rate, debug=False)
    vae_init.train(t, d_flat, batch_size=batch_size, training_epochs=1)

    vae = VaeNormalFit(model, mode_corr='infer_post_corr', learning_rate=learning_rate, debug=False, vae_init=vae_init)
    time, ret = runtime(vae.train, t, d_flat, batch_size=batch_size, training_epochs=epochs)
    
    # Create output directory
    makedirs(outdir, exist_ok=True)

    # Dump cost history
    cost_history = ret[1]
    with open(os.path.join(outdir, "cost_history.tsv"), "w") as f:
        for idx, cost in enumerate(cost_history):
            approx_time = idx * time / epochs 
            f.write("%i\t%f\t%f\n" % (idx, approx_time, cost))

    # Transpose output so the first index is by parameter
    # FIXME most of the below should be built in to the inference engine
    means = vae.output("mp_mean_model")
    variances = np.transpose(np.diagonal(vae.output("mp_covar"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    for idx, param in enumerate(model.params):
        if mask is not None:
            mean = np.zeros(shape[:3])
            mean[d_mask > 0] = means[idx]
            variance = np.zeros(shape[:3])
            variance[d_mask > 0] = variances[idx]
        else:
            mean = means[idx].reshape(shape[:3])
            variance = variances[idx].reshape(shape[:3])

        nii_mean = nib.Nifti1Image(mean, None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = nib.Nifti1Image(variance, None, header=nii.get_header())
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out modelfit
    # FIXME have to tile parameters because require 1 value per time point in evaluate
    tiled_params = np.tile(np.expand_dims(means, axis=-1), (1, 1, shape[3]))
    fit = model.ievaluate(tiled_params, t)
    if mask is not None:
        mfit = np.zeros(shape)
        mfit[d_mask > 0] = fit
    else:
        mfit = fit.reshape(shape)

    fit_nii = nib.Nifti1Image(mfit, None, header=nii.get_header())
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

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

if __name__ == "__main__":
    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    test_asl(fname="mpld_asltc_diff.nii", 
             mask="mpld_asltc_mask.nii", 
             plds=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 
             slicedt=0.0452,
             repeats=8, 
             tau=1.8, 
             outdir="asl", 
             epochs=100, 
             batch_size=8)

