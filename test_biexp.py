import os

import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from exp_models import ExpModel, BiExpModel
from inference import VaeNormalFit

def test_biexp(fname, t0, dt, outdir=".", learning_rate=0.02, batch_size=10, epochs=100, **kwargs):
    """
    Fit to a 4D Nifti image

    :param fname: File name of Nifti image
    :param mode: Model to fit to
    :param t0: Timeseries first value
    :param dt: Time step
    """
    nii = nib.load(fname)
    d = nii.get_data()
    shape = d.shape
    d_flat = d.reshape(-1, shape[-1])

    model = BiExpModel()

    # Generate timeseries FIXME should be derivable from the model
    t = np.linspace(t0, t0+shape[3]*dt, num=shape[3], endpoint=False)

    # Train with no correlation between parameters
    vae = VaeNormalFit(model, mode_corr='no_post_corr', learning_rate=learning_rate, draw_size=batch_size)
    time, ret = runtime(vae.train, t, d_flat, batch_size=batch_size, training_epochs=epochs)
    cost_history = ret[1]
    for idx, cost in enumerate(cost_history):
        ct = idx * time / epochs 
        print("%f\t%f" % (ct, cost))

    # Transpose output so the first index is by parameter
    means = vae.output("mp_mean_model")
    variances = np.transpose(np.diagonal(vae.output("mp_covar"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    makedirs(outdir, exist_ok=True)
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(means[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = nib.Nifti1Image(variances[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out modelfit
    # FIXME currently have to tile parameters because require 1 value per time point in evaluate
    tiled_params = np.tile(np.expand_dims(means, axis=-1), (1, 1, len(t)))
    fit = model.ievaluate(tiled_params, t)
    fit_nii = nib.Nifti1Image(fit.reshape(shape), None, header=nii.get_header())
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

def runtime(runnable, *args, **kwargs):
    import time
    t0 = time.time()
    ret = runnable(*args, **kwargs)
    t1 = time.time()
    return (t1-t0), ret

def makedirs(d, exist_ok=False):
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

    test_biexp("test_data_exp.nii", 0, 0.02, "noisy")

