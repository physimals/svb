import os
import math

import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from svb.exp_models import ExpModel, BiExpModel
from svb.svb import SvbFit
from svb.dist import FoldedNormal, Normal

def test_biexp_single():
    # Ground truth
    n_samples=200
    true_amp1 = 10.0 # the true amplitude of the model
    true_amp2 = 10.0
    true_R1 = 1.0 # the true decay rate
    true_R2 = 10.0
    true_sd = 1e0 # the s.d. of the noise
    true_var = true_sd*true_sd

    # Time points
    t_end = 5*1/true_R1 #for time being, make sure we have a good range of samples based on the choice of R1
    t = np.linspace(0,t_end,num=n_samples)

    param_overrides = {
    #    "amp1" : {"dist" : FoldedNormal(1.0, 1e6), "log_var_init" : 1.0},
    #    "r1" : {"dist" : FoldedNormal(1.0, 1e6), "log_var_init" : -1.0},
    #    "amp2" : {"dist" : FoldedNormal(1.0, 1e6), "log_var_init" : 1.0},
    #    "r2" : {"dist" : FoldedNormal(1.0, 1e6), "log_var_init" : -1.0},
    }
    model = BiExpModel(param_overrides=param_overrides)
   
    shape = (1, n_samples)
    y_true = np.zeros(shape)
    for t_idx in range(n_samples):
        y_true[..., t_idx] = true_amp1*math.exp(-true_R1*t[t_idx]) + true_amp2*math.exp(-true_R2*t[t_idx])

    # Add noise
    x = np.random.normal(y_true, true_sd)
    t = np.reshape(t,(1,-1))

    # Learning parameters
    learning_rate = 0.01
    batch_size = 20
    training_epochs = 500
    svb = SvbFit(model, infer_covar=False, debug=False)
    time, ret = runtime(svb.train, t, x, 
                        batch_size=batch_size, 
                        training_epochs=training_epochs, 
                        learning_rate=learning_rate, 
                        fit_only_epochs=200)

    # Transpose output so the first index is by parameter
    means = svb.output("model_params")
    print(means)
    variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

    # Plot the estimated functions overlaid on data
    plt.figure()
    plt.cla()

    # Plot the data
    plt.plot(t[0], x[0], 'rx')

    # Plot the ground truth
    plt.plot(t[0], y_true[0], 'g-')

    # Plot the fit using the inital guess (for reference)
    #mn = svb_norm_init.sess.run(svb_norm_init.mp_mean)
    #y_est = sess.run(nlinmod.evaluate(np.reshape(mn[0,0:nlinmod.nparams],[-1,1]),t))
    #plt.plot(t,y_est,'k.')
    
    # Plot the fit with the estimated parameter values
    fit = ret[4]
    plt.plot(t[0], fit[0], 'b')

    plt.show()

def test_biexp(fname, t0, dt, outdir=".", 
               learning_rate=0.2, batch_size=10, epochs=500, mask=None, **kwargs):
    """
    Fit to a 4D Nifti image

    :param fname: File name of Nifti image
    :param mode: Model to fit to
    :param t0: Timeseries first value
    :param dt: Time step
    """
    nii = nib.load(fname)
    d = nii.get_data()
    shape = list(d.shape)
    d_flat = d.reshape(-1, shape[-1])

    if mask is not None:
        mask_nii = nib.load(mask)
        mask_d = mask_nii.get_data()
        mask_flat = mask_d.flatten()
        d_flat = d_flat[mask_flat > 0]

    model = BiExpModel()

    # Generate timeseries FIXME should be derivable from the model
    t = np.linspace(t0, t0+shape[3]*dt, num=shape[3], endpoint=False)

    # Train with no correlation between parameters
    svb = SvbFit(model, infer_covar=False, debug=False)
    time, ret = runtime(svb.train, t, d_flat, 
                        batch_size=batch_size, 
                        training_epochs=epochs, 
                        max_trials=50, 
                        learning_rate=learning_rate)
    #avg_cost_history = ret[0]
    #for idx, cost in enumerate(avg_cost_history):
    #    ct = idx * time / epochs 
    #    print("%f\t%f" % (ct, cost))

    # Transpose output so the first index is by parameter
    means = svb.output("model_params")
    variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    makedirs(outdir, exist_ok=True)
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(means[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = nib.Nifti1Image(variances[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out modelfit
    fit = ret[4]
    fit_nii = nib.Nifti1Image(fit.reshape(shape), None, header=nii.get_header())
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

    # Write out voxelwise cost history
    cost_history_v = ret[2]
    cost_history_v_nii = nib.Nifti1Image(cost_history_v.reshape(shape[:3] + [cost_history_v.shape[1]]), None, header=nii.get_header())
    cost_history_v_nii.to_filename(os.path.join(outdir, "cost_history.nii.gz"))

    # Write out voxelwise parameter history
    param_history_v = ret[3]
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(param_history_v[:, :, idx].reshape(list(shape[:3]) + [cost_history_v.shape[1]]), None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s_history.nii.gz" % param.name))

    nii_mean = nib.Nifti1Image(param_history_v[:, :, model.nparams].reshape(list(shape[:3]) + [cost_history_v.shape[1]]), None, header=nii.get_header())
    nii_mean.to_filename(os.path.join(outdir, "mean_noise_history.nii.gz"))

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
    import logging.config
    if os.path.exists("logging.conf"):
        logging.config.fileConfig("logging.conf")

    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    test_biexp_single()
    #test_biexp("test_data_exp.nii", t0=0, dt=0.02, outdir="noisy", epochs=300, learning_rate=0.05)

